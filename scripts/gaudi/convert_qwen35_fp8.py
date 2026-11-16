"""Convert Qwen3.5 language Linear weights to portable, Gaudi2-safe E4M3.

Embedding, vision, recurrent gates, convolutions, norms and MTP retain their
source precision. INT8 ConvRot sources are dequantized in FP32 before FP8
quantization; protected quantized tensors are stored in BF16.
"""
import argparse
import json
import os
from pathlib import Path
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE, dequantize_int8_convrot_weight_dtype


def convert(source, destination, *, from_int8_convrot=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    output, layers = {}, {}
    started = time.monotonic()
    with safe_open(str(source), framework='pt', device='cpu') as handle:
        metadata = handle.metadata() or {}
        configs = {}
        removed = set()
        quant_keys = [key for key in handle.keys() if key.endswith('.comfy_quant')]
        if '_quantization_metadata' in metadata or (quant_keys and not from_int8_convrot):
            raise ValueError('Use --from-int8-convrot for inline INT8 ConvRot metadata, or convert source weights to floating point first')
        if from_int8_convrot:
            if not quant_keys:
                raise ValueError('Expected an INT8 ConvRot checkpoint with inline comfy_quant metadata')
            for key in quant_keys:
                prefix = key.removesuffix('.comfy_quant')
                config = json.loads(handle.get_tensor(key).numpy().tobytes().decode('utf-8'))
                if config.get('format') != 'int8_tensorwise' or not config.get('convrot', False):
                    raise ValueError(f'Unsupported quantization for {prefix}: {config}')
                configs[prefix] = config
                removed.update((key, prefix + '.weight_scale'))
        if 'model.language_model.layers.0.linear_attn.in_proj_qkv.weight' not in handle.keys():
            raise ValueError('Expected a Qwen3.5 ComfyUI checkpoint')
        for key in handle.keys():
            if key in removed:
                continue
            value = handle.get_tensor(key)
            language_linear = key.startswith('model.language_model.layers.') and value.ndim == 2 and key.endswith('.weight')
            gate = key.endswith(('.in_proj_a.weight', '.in_proj_b.weight'))
            quantize = (language_linear and not gate) or key == 'lm_head.weight'
            prefix = key.removesuffix('.weight') if key.endswith('.weight') else None
            if prefix in configs:
                config = configs[prefix]
                if value.dtype != torch.int8:
                    raise ValueError(f'Expected INT8 weight for {key}')
                value = dequantize_int8_convrot_weight_dtype(
                    value, handle.get_tensor(prefix + '.weight_scale').float(),
                    int(config.get('convrot_groupsize', 256)), DTYPE_TO_CODE[torch.float32])
                if value.dtype != torch.float32 or not value.isfinite().all():
                    raise ValueError(f'Invalid FP32 dequantized weight {key}')
                if not quantize:
                    value = value.to(torch.bfloat16)
            if quantize:
                if value.dtype not in (torch.float32, torch.bfloat16, torch.float16):
                    raise ValueError(f'Expected floating-point weight for {key}')
                value = value.float()
                if not value.isfinite().all():
                    raise ValueError(f'Nonfinite source weight {key}')
                scale = value.abs().amax().clamp_min(1e-12) / 224.0
                output[key] = (value / scale).to(torch.float8_e4m3fn).contiguous()
                output[key[:-len('.weight')] + '.weight_scale'] = scale
                layers[key[:-len('.weight')]] = {'format': 'float8_e4m3fn'}
            else:
                output[key] = value.contiguous()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.part-{os.getpid()}')
    try:
        save_file(output, str(temporary), metadata={
            'format': 'pt', 'source': source.name,
            'source_conversion': 'int8_convrot_to_fp32' if from_int8_convrot else metadata.get('conversion', ''),
            'conversion': 'qwen35_language_linear_e4m3_range224',
            '_quantization_metadata': json.dumps({'layers': layers}),
        })
        # Atomic publication also prevents concurrent converters overwriting a file.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {'source': str(source), 'destination': str(destination), 'converted_linears': len(layers),
            'bytes': destination.stat().st_size, 'seconds': time.monotonic() - started}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--from-int8-convrot', action='store_true', help='Dequantize INT8 ConvRot weights in FP32 before conversion')
    args = parser.parse_args()
    torch.set_num_threads(16)
    print(json.dumps(convert(args.source, args.destination, from_int8_convrot=args.from_int8_convrot), indent=2))
