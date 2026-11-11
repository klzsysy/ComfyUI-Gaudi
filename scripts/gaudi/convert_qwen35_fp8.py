"""Convert Qwen3.5 language Linear weights to portable, Gaudi2-safe E4M3.

Embedding, vision, recurrent gates, convolutions, norms and MTP retain their
source precision. Accepts a floating-point ComfyUI single-file checkpoint.
"""
import argparse
import json
import os
from pathlib import Path
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def convert(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    output, layers = {}, {}
    started = time.monotonic()
    with safe_open(str(source), framework='pt', device='cpu') as handle:
        metadata = handle.metadata() or {}
        if '_quantization_metadata' in metadata or any(key.endswith('.comfy_quant') for key in handle.keys()):
            raise ValueError('Convert quantized source weights to floating point first')
        if 'model.language_model.layers.0.linear_attn.in_proj_qkv.weight' not in handle.keys():
            raise ValueError('Expected a Qwen3.5 ComfyUI checkpoint')
        for key in handle.keys():
            value = handle.get_tensor(key)
            language_linear = key.startswith('model.language_model.layers.') and value.ndim == 2 and key.endswith('.weight')
            gate = key.endswith(('.in_proj_a.weight', '.in_proj_b.weight'))
            if (language_linear and not gate) or key == 'lm_head.weight':
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
            'source_conversion': metadata.get('conversion', ''),
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
    args = parser.parse_args()
    torch.set_num_threads(16)
    print(json.dumps(convert(args.source, args.destination), indent=2))
