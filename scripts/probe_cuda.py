#!/usr/bin/env python3
"""Print a compact PyTorch CUDA environment probe."""

import json
import os

import torch

result = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}
if result["device_count"]:
    result["device_name"] = torch.cuda.get_device_name(0)
    result["capability"] = list(torch.cuda.get_device_capability(0))
print(json.dumps(result))
