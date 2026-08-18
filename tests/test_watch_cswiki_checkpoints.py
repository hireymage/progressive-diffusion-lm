from types import SimpleNamespace

import scripts.watch_cswiki_checkpoints as watch


def test_watch_runs_batch_once_for_new_checkpoint(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(watch, "checkpoint_ids",
                        lambda _dir, _include: [("step_0010000", 10000)])
    monkeypatch.setattr(watch, "run_batch",
                        lambda args, checkpoint_dir, output_dir: calls.append((args.name, checkpoint_dir, output_dir)))
    monkeypatch.setattr(watch.time, "sleep", lambda _seconds: None)
    args = SimpleNamespace(
        name="m1-256",
        cache_dir=tmp_path / "cache",
        checkpoint_dir=tmp_path / "checkpoints",
        models_json=tmp_path / "models.json",
        output_root=tmp_path / "out",
        prompts=2,
        max_new_tokens=4,
        routes=["q8_only", "q2_q8_fp16"],
        measure_exits=True,
        include_latest_best=True,
        poll_seconds=1,
        once=True,
    )
    summary = watch.watch(args)
    assert len(calls) == 1
    assert summary["observed_checkpoints"] == [["step_0010000", 10000]]
    assert summary["batches"][0]["checkpoints"] == [["step_0010000", 10000]]


def test_parser_exposes_watch_arguments():
    parsed = watch.parser()
    assert "--name" in parsed._option_string_actions
    assert "--poll-seconds" in parsed._option_string_actions
