"""Focused orchestration tests; these never import MLX or launch training."""
from __future__ import annotations

import datetime as dt
import csv
import importlib.util
import json
import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_dual_m1_campaign.py"
SPEC = importlib.util.spec_from_file_location("dual_m1_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


class CampaignHarnessTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "campaign_slug": "paired-replication",
            "node": "m1-512",
            "seeds": [31415, 27182],
            "max_parallel_tasks": 1,
            "tasks": [
                {"id": "first", "description": "first", "command": ["{python}", "noop.py"]},
                {"id": "second", "description": "second", "command": ["{python}", "noop.py"]},
            ],
        }

    def test_run_ids_are_safe_and_unique(self):
        now = dt.datetime(2026, 7, 18, 12, 34, 56, tzinfo=dt.timezone.utc)
        one = campaign.new_run_id(self.config, now=now, nonce="deadbeef")
        two = campaign.new_run_id(self.config, now=now, nonce="cafebabe")
        self.assertEqual(one, "20260718-123456_paired-replication_s31415-27182_deadbeef")
        self.assertNotEqual(one, two)
        campaign.validate_run_id(one)
        with self.assertRaises(campaign.CampaignError):
            campaign.validate_run_id("../shared")

    def test_rejects_wrong_or_unknown_node_and_parallel_config(self):
        with self.assertRaises(campaign.CampaignError):
            campaign.validate_node("m1-256", self.config)
        with self.assertRaises(campaign.CampaignError):
            campaign.validate_node("m1-999", self.config)
        bad = dict(self.config, max_parallel_tasks=2)
        with self.assertRaises(campaign.CampaignError):
            campaign.validate_node("m1-512", bad)

    def test_rejects_runtime_inside_source_and_wrong_shared_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "project"
            (source / "src").mkdir(parents=True)
            with self.assertRaises(campaign.CampaignError):
                campaign.ensure_safe_paths(source, source / "runtime", source / "results")
            with self.assertRaises(campaign.CampaignError):
                campaign.ensure_safe_paths(source, root / "local", root / "other-results")
            campaign.ensure_safe_paths(source, root / "local", source / "results")

    def test_resume_state_skips_only_successful_completed_tasks(self):
        state = campaign.initial_state(self.config, "20260718-123456_paired-replication_s31415-27182_deadbeef")
        self.assertFalse(campaign.task_is_complete(state, "first"))
        state["tasks"]["first"].update(status="completed", exit_code=0)
        state["tasks"]["second"].update(status="failed", exit_code=9)
        self.assertTrue(campaign.task_is_complete(state, "first"))
        self.assertFalse(campaign.task_is_complete(state, "second"))

    def test_config_materialization_isolated_and_order_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            base = repo / "configs/base.json"
            base.parent.mkdir(parents=True)
            base.write_text(json.dumps({"train": {"seed": 1, "results_dir": "results"}, "model": {"bits": [16]}}))
            task = {
                "id": "first",
                "base_config": "configs/base.json",
                "overrides": {"train": {"seed": 99}, "model": {"bits": [1]}},
            }
            generated = campaign.materialize_task_config(repo, task)
            value = json.loads((repo / generated).read_text())
            self.assertEqual(value["train"], {"seed": 99, "results_dir": "results"})
            self.assertEqual(value["model"]["bits"], [1])
            self.assertEqual([t["id"] for t in self.config["tasks"]], ["first", "second"])

    def test_publish_is_unique_read_only_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "manifest.json").write_text("{}\n")
            shared = root / "results"
            run_id = "20260718-123456_paired-replication_s31415-27182_deadbeef"
            destination = campaign.publish_bundle(bundle, shared, "m1-512", run_id)
            self.assertTrue((destination / "manifest.json").exists())
            self.assertEqual((destination / "manifest.json").stat().st_mode & 0o222, 0)
            with self.assertRaises(campaign.CampaignError):
                campaign.publish_bundle(bundle, shared, "m1-512", run_id)

    def test_resume_of_verified_published_run_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "project"
            (source / "src").mkdir(parents=True)
            config_path = root / "campaign.json"
            config_path.write_text(json.dumps(self.config))
            run_id = "20260718-123456_paired-replication_s31415-27182_deadbeef"
            published = source / "results/m1-512" / run_id
            published.mkdir(parents=True)
            (published / "manifest.json").write_text(json.dumps({"run_id": run_id}))
            args = Namespace(
                source=source,
                campaign=config_path,
                node="m1-512",
                local_root=root / "local",
                shared_results_root=source / "results",
                run_id=run_id,
                resume=True,
                dry_run=False,
            )
            self.assertEqual(campaign.execute(args), 0)
    def test_matched_noise_campaign_configs_have_exact_paired_matrix(self):
        project = SCRIPT.parents[1]
        expectations = {"m1-256": [11, 29], "m1-512": [47, 73]}
        for node, seeds in expectations.items():
            value = json.loads(
                (project / f"configs/campaign/{node}-matched-noise.json").read_text()
            )
            self.assertEqual(value["node"], node)
            self.assertEqual(value["seeds"], seeds)
            self.assertEqual(value["max_parallel_tasks"], 1)
            self.assertEqual(len(value["tasks"]), 8)
            self.assertEqual(len({task["id"] for task in value["tasks"]}), 8)
            self.assertEqual(
                len({task["overrides"]["experiment_name"] for task in value["tasks"]}), 8
            )
            observed = []
            for task in value["tasks"]:
                model = task["overrides"]["model"]
                train = task["overrides"]["train"]
                observed.append((train["seed"], model["model_type"], model["weight_noise_mode"]))
                self.assertEqual(train["max_steps"], 10_000)
                self.assertFalse(train["save_checkpoints"])
                self.assertIn("expected_metrics_contract", task)
                if model["model_type"] == "progressive":
                    self.assertEqual(model["precision_schedule"], [1] * 8)
                    self.assertEqual(model["weight_noise_mode"], "none")
                else:
                    self.assertEqual(model["precision_schedule"], [16] * 8)
            expected = [
                (seed, model_type, noise_mode)
                for seed in seeds
                for model_type, noise_mode in [
                    ("baseline", "none"),
                    ("progressive", "none"),
                    ("baseline", "gaussian_matched"),
                    ("baseline", "uniform_matched"),
                ]
            ]
            self.assertEqual(observed, expected)

    def test_semantic_metrics_contract_accepts_complete_finite_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            task = self._write_metrics_fixture(repo)
            campaign.validate_task_completion(repo, task, 0)

    def test_zero_exit_does_not_complete_missing_or_nonfinite_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            task = self._write_metrics_fixture(repo)
            metrics = repo / task["expected_metrics_contract"]["artifact_dir"] / "train_metrics.csv"
            metrics.write_text(metrics.read_text().replace("1.25", "nan"))
            with self.assertRaisesRegex(campaign.CampaignError, "nonfinite"):
                campaign.validate_task_completion(repo, task, 0)

            metrics.unlink()
            with self.assertRaisesRegex(campaign.CampaignError, "missing"):
                campaign.validate_task_completion(repo, task, 0)

    def test_zero_exit_does_not_complete_wrong_final_summary_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            task = self._write_metrics_fixture(repo)
            summary_path = repo / task["expected_metrics_contract"]["artifact_dir"] / "final_summary.json"
            summary = json.loads(summary_path.read_text())
            summary["seed"] = 12
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(campaign.CampaignError, "seed"):
                campaign.validate_task_completion(repo, task, 0)

    def test_legacy_task_without_contract_remains_exit_code_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign.validate_task_completion(Path(tmp), self.config["tasks"][0], 0)

    def _write_metrics_fixture(self, repo: Path):
        artifact_dir = repo / "results/campaign/exp"
        artifact_dir.mkdir(parents=True)
        train_fields = [
            "step", "train_loss", "gradient_norm", "q1_residual_rms",
            "injected_noise_rms", "bits_used", "lr", "elapsed_s",
        ]
        with (artifact_dir / "train_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=train_fields)
            writer.writeheader()
            writer.writerow({
                "step": 10, "train_loss": 2.5, "gradient_norm": 1.25,
                "q1_residual_rms": 0.1, "injected_noise_rms": 0.1,
                "bits_used": 16, "lr": 0.001, "elapsed_s": 1.0,
            })
        eval_fields = [
            "step", "val_loss", "val_perplexity", "val_accuracy",
            "generalization_gap", "train_loss", "bits_used",
        ]
        (artifact_dir / "eval_history.json").write_text(json.dumps([{
            "step": 10, "val_loss": 2.7, "val_perplexity": math.exp(2.7),
            "val_accuracy": 0.2, "generalization_gap": 0.2,
            "train_loss": 2.5, "bits_used": 16,
        }]))
        (artifact_dir / "final_summary.json").write_text(json.dumps({
            "experiment_name": "exp", "model_type": "baseline",
            "weight_noise_mode": "gaussian_matched", "seed": 11,
            "best_val_loss": 2.7, "total_training_seconds": 1.0,
        }))
        return {
            "id": "exp-s11",
            "expected_metrics_contract": {
                "artifact_dir": "results/campaign/exp",
                "train_metrics_columns": train_fields,
                "eval_history_fields": eval_fields,
                "finite_summary_fields": ["best_val_loss", "total_training_seconds"],
                "expected_summary": {
                    "experiment_name": "exp",
                    "model_type": "baseline",
                    "weight_noise_mode": "gaussian_matched",
                    "seed": 11,
                },
            },
        }


if __name__ == "__main__":
    unittest.main()
