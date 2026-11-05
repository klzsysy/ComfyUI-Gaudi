"""Run from the repo root with eager mode and GC_KERNEL_PATH set (GC_KERNEL_PATH must point to libtpc_kernels.so)."""
import json

import torch

from comfy import model_management as mm
from comfy.ldm.modules.attention import attention_pytorch
from comfy.model_patcher import ModelPatcher


def main():
    device = mm.get_torch_device()
    assert device.type == "hpu", device
    assert device in mm.get_all_torch_devices()
    assert mm.should_use_bf16(device) and not mm.should_use_fp16(device)
    assert not mm.supports_fp64(device)
    assert not mm.supports_int8_compute(device)
    assert not mm.supports_fp8_compute(device)
    assert mm.DISABLE_PINNED_MEMORY and mm.get_offload_stream(device) is None
    mm.set_torch_device(device)
    total, used = mm.get_total_memory(device, True)
    free, cache = mm.get_free_memory(device, True)
    assert 0 < free <= total and 0 <= used < total and cache == 0

    torch.manual_seed(42)
    x = torch.randn(64, 64, dtype=torch.bfloat16)
    actual = x.to(device) @ x.to(device)
    assert actual.device.type == "hpu"
    torch.testing.assert_close(actual.cpu().float(), x.float() @ x.float(), atol=0.15, rtol=0.02)

    q, k, v = [torch.randn(1, 32, 128, dtype=torch.bfloat16) for _ in range(3)]
    expected = attention_pytorch(q.float(), k.float(), v.float(), heads=2)
    actual = attention_pytorch(q.to(device), k.to(device), v.to(device), heads=2)
    assert actual.device.type == "hpu"
    torch.testing.assert_close(actual.cpu().float(), expected, atol=0.025, rtol=0.03)

    model = torch.nn.Linear(64, 64, dtype=torch.bfloat16)
    expected = model(x).detach()
    patcher = ModelPatcher(model, load_device=device, offload_device=torch.device("cpu"))
    mm.load_models_gpu([patcher])
    assert model.weight.device.type == "hpu"
    torch.testing.assert_close(model(x.to(device)).detach().cpu(), expected, atol=0.02, rtol=0.02)
    mm.unload_all_models()
    assert model.weight.device.type == "cpu"
    mm.synchronize()
    mm.soft_empty_cache()
    print(json.dumps({"device": mm.get_torch_device_name(device), "torch": torch.__version__,
                      "total_bytes": total, "free_bytes": free,
                      "checks": ["bf16_matmul", "comfy_attention", "model_load_unload", "memory", "synchronize"],
                      "result": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
