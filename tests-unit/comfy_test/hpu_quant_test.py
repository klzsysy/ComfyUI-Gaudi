from dataclasses import replace

import pytest
import torch

from comfy.quant_ops import TensorCoreFP8Layout


@pytest.mark.skipif(not hasattr(torch, "hpu") or not torch.hpu.is_available(), reason="Requires an HPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_hpu_e4m3fn_full_range(dtype):
    # Include both signs, subnormals, signed zero, values > 240, and NaN.
    qdata = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(16, 16)
    params = TensorCoreFP8Layout.Params(scale=torch.tensor(0.01), orig_dtype=dtype, orig_shape=qdata.shape)
    expected = TensorCoreFP8Layout.dequantize(qdata, params)
    actual = TensorCoreFP8Layout.dequantize(qdata.to("hpu"), replace(params, scale=params.scale.to("hpu")))
    assert actual.device.type == "hpu"
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=True)
