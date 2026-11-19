"""Convert Qwen3.5 language Linear weights to portable, Gaudi2-safe E4M3.

Embedding, vision, recurrent gates, convolutions and norms retain their
source precision. MTP weights retain source precision by default and can
optionally be attached from another Qwen3.5 checkpoint and/or quantized to
E4M3. INT8 ConvRot sources are dequantized in FP32 before FP8 quantization;
protected quantized tensors are stored in BF16.
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


def _read_convrot_configs(handle, keys):
    configs, removed = {}, set()
    for key in keys:
        if not key.endswith('.comfy_quant'):
            continue
        prefix = key.removesuffix('.comfy_quant')
        config = json.loads(handle.get_tensor(key).numpy().tobytes().decode('utf-8'))
        if config.get('format') != 'int8_tensorwise' or not config.get('convrot', False):
            raise ValueError(f'Unsupported quantization for {prefix}: {config}')
        configs[prefix] = config
        removed.update((key, prefix + '.weight_scale'))
    return configs, removed


def _dequantize_if_convrot(handle, key, value, configs, quantize):
    prefix = key.removesuffix('.weight') if key.endswith('.weight') else None
    if prefix not in configs:
        return value
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
    return value


def _store_tensor(output, layers, key, value, quantize):
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
        if value.is_floating_point() and not value.isfinite().all():
            raise ValueError(f'Nonfinite source weight {key}')
        output[key] = value.contiguous()


def convert(source, destination, *, from_int8_convrot=False, attach_mtp=None, quantize_mtp=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    mtp_path = Path(attach_mtp).resolve() if attach_mtp is not None else None
    output, layers = {}, {}
    mtp_tensors = 0
    mtp_converted = 0
    started = time.monotonic()
    with safe_open(str(source), framework='pt', device='cpu') as handle:
        metadata = handle.metadata() or {}
        source_keys = list(handle.keys())
        quant_keys = [key for key in source_keys if key.endswith('.comfy_quant')]
        if '_quantization_metadata' in metadata or (quant_keys and not from_int8_convrot):
            raise ValueError('Use --from-int8-convrot for inline INT8 ConvRot metadata, or convert source weights to floating point first')
        configs, removed = ({}, set())
        if from_int8_convrot:
            if not quant_keys:
                raise ValueError('Expected an INT8 ConvRot checkpoint with inline comfy_quant metadata')
            configs, removed = _read_convrot_configs(handle, quant_keys)
        if 'model.language_model.layers.0.linear_attn.in_proj_qkv.weight' not in source_keys:
            raise ValueError('Expected a Qwen3.5 ComfyUI checkpoint')
        if mtp_path is not None and any(key.startswith('mtp.') for key in source_keys):
            raise ValueError('Source checkpoint already contains mtp.* tensors; omit --attach-mtp')
        for key in source_keys:
            if key in removed:
                continue
            value = handle.get_tensor(key)
            language_linear = key.startswith('model.language_model.layers.') and value.ndim == 2 and key.endswith('.weight')
            gate = key.endswith(('.in_proj_a.weight', '.in_proj_b.weight'))
            mtp_linear = quantize_mtp and key.startswith('mtp.') and value.ndim == 2 and key.endswith('.weight')
            quantize = (language_linear and not gate) or key == 'lm_head.weight' or mtp_linear
            value = _dequantize_if_convrot(handle, key, value, configs, quantize)
            _store_tensor(output, layers, key, value, quantize)
            if key.startswith('mtp.'):
                mtp_tensors += 1
                if quantize:
                    mtp_converted += 1

    mtp_source_conversion = ''
    if mtp_path is not None:
        with safe_open(str(mtp_path), framework='pt', device='cpu') as mtp_handle:
            mtp_meta = mtp_handle.metadata() or {}
            mtp_keys = [key for key in mtp_handle.keys() if key.startswith('mtp.')]
            if 'mtp.fc.weight' not in mtp_keys:
                raise ValueError(f'Expected mtp.* tensors in {mtp_path}')
            mtp_quant_meta = json.loads(mtp_meta['_quantization_metadata']).get('layers', {}) if '_quantization_metadata' in mtp_meta else {}
            if any(k.startswith('mtp.') for k in mtp_quant_meta):
                raise ValueError('MTP source checkpoint already has quantized mtp.* metadata; use a floating-point or INT8 ConvRot MTP source')
            mtp_quant_keys = [key for key in mtp_keys if key.endswith('.comfy_quant')]
            mtp_configs, mtp_removed = _read_convrot_configs(mtp_handle, mtp_quant_keys) if mtp_quant_keys else ({}, set())
            mtp_source_conversion = 'int8_convrot_to_fp32' if mtp_configs else mtp_meta.get('conversion', '')
            for key in mtp_keys:
                if key in mtp_removed:
                    continue
                value = mtp_handle.get_tensor(key)
                mtp_linear = quantize_mtp and value.ndim == 2 and key.endswith('.weight')
                value = _dequantize_if_convrot(mtp_handle, key, value, mtp_configs, mtp_linear)
                _store_tensor(output, layers, key, value, mtp_linear)
                mtp_tensors += 1
                if mtp_linear:
                    mtp_converted += 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.part-{os.getpid()}')
    out_metadata = {
        'format': 'pt', 'source': source.name,
        'source_conversion': 'int8_convrot_to_fp32' if from_int8_convrot else metadata.get('conversion', ''),
        'conversion': 'qwen35_language_linear_e4m3_range224',
        '_quantization_metadata': json.dumps({'layers': layers}),
    }
    if mtp_path is not None:
        out_metadata['mtp_source'] = mtp_path.name
        out_metadata['mtp_source_conversion'] = mtp_source_conversion
    if quantize_mtp:
        out_metadata['mtp_quantization'] = 'e4m3_range224'
    try:
        save_file(output, str(temporary), metadata=out_metadata)
        # Atomic publication also prevents concurrent converters overwriting a file.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {'source': str(source), 'destination': str(destination), 'converted_linears': len(layers),
            'mtp_source': str(mtp_path) if mtp_path is not None else None,
            'mtp_tensors': mtp_tensors, 'mtp_converted_linears': mtp_converted,
            'bytes': destination.stat().st_size, 'seconds': time.monotonic() - started}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--from-int8-convrot', action='store_true', help='Dequantize INT8 ConvRot weights in FP32 before conversion')
    parser.add_argument('--attach-mtp', type=Path, default=None, help='Attach mtp.* tensors from another Qwen3.5 checkpoint')
    parser.add_argument('--quantize-mtp', action='store_true', help='Also quantize 2D mtp.*.weight Linear layers to E4M3')
    args = parser.parse_args()
    torch.set_num_threads(16)
    print(json.dumps(convert(args.source, args.destination, from_int8_convrot=args.from_int8_convrot,
                             attach_mtp=args.attach_mtp, quantize_mtp=args.quantize_mtp), indent=2))
