"""Convert MLX flexible checkpoints to resumable PyTorch checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .torch_layerwise_model import TorchLayerwiseConfig, TorchLayerwiseProgressiveLM


class MLXCompatibleAdamW(torch.optim.Optimizer):
    """AdamW with MLX defaults, notably no bias correction."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("MLXCompatibleAdamW does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                state["exp_avg"].mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                denominator = state["exp_avg_sq"].sqrt().add_(group["eps"])
                parameter.addcdiv_(state["exp_avg"], denominator, value=-group["lr"])
        return loss


def convert_mlx_checkpoint(path: Path, device: torch.device, lr: float = 1e-3):
    metadata = json.loads(path.with_suffix(".json").read_text())
    architecture = metadata.get("architecture")
    if not isinstance(architecture, list) or len(architecture) != 5:
        raise ValueError("MLX checkpoint is missing the architecture contract")
    n_layers, d_model, d_ff, n_heads, seq_len = map(int, architecture)
    with np.load(path) as payload:
        vocab_size = int(payload["token_embed.weight"].shape[0] - 1)
        cfg = TorchLayerwiseConfig(vocab_size, d_model, d_ff, n_heads, n_layers, seq_len)
        model = TorchLayerwiseProgressiveLM(cfg).to(device)
        state = model.state_dict()
        missing = sorted(set(state) - {key for key in payload.files if not key.startswith("opt_")})
        unexpected = sorted({key for key in payload.files if not key.startswith("opt_")} - set(state))
        if missing or unexpected:
            raise ValueError(f"weight key mismatch; missing={missing}, unexpected={unexpected}")
        model.load_state_dict({key: torch.from_numpy(np.array(payload[key])).to(device) for key in state})
        optimizer = MLXCompatibleAdamW(model.parameters(), lr=lr)
        global_step = int(np.asarray(payload["opt_step"]))
        for name, parameter in model.named_parameters():
            m_key, v_key = f"opt_{name}.m", f"opt_{name}.v"
            if m_key not in payload or v_key not in payload:
                raise ValueError(f"optimizer state missing for {name}")
            optimizer.state[parameter] = {
                "step": global_step,
                "exp_avg": torch.from_numpy(np.array(payload[m_key])).to(device),
                "exp_avg_sq": torch.from_numpy(np.array(payload[v_key])).to(device),
            }
    return model, optimizer, metadata


def save_torch_checkpoint(path: Path, model, optimizer, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.pt")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "metadata": metadata}, temporary)
    temporary.replace(path)
    path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


def load_torch_checkpoint(path: Path, device: torch.device, lr: float = 1e-3):
    payload = torch.load(path, map_location=device, weights_only=False)
    metadata = payload["metadata"]
    n_layers, d_model, d_ff, n_heads, seq_len = map(int, metadata["architecture"])
    vocab_size = int(payload["model"]["token_embed.weight"].shape[0] - 1)
    model = TorchLayerwiseProgressiveLM(TorchLayerwiseConfig(
        vocab_size, d_model, d_ff, n_heads, n_layers, seq_len)).to(device)
    model.load_state_dict(payload["model"])
    optimizer = MLXCompatibleAdamW(model.parameters(), lr=lr)
    optimizer.load_state_dict(payload["optimizer"])
    return model, optimizer, metadata
