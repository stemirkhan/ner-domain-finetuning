"""Explicit encoder-linear NF4 conversion for the frozen GLiNER2.5 base.

GLiNER's loader is not transformers.PreTrainedModel.from_pretrained and rejects
quantization_config. Embeddings, norms and task heads remain FP32. Linear biases
start FP32; bitsandbytes casts them to BF16 on forward. This module does not attach
adapters, train, merge, or save a quantized checkpoint.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

BITSANDBYTES_VERSION = '0.50.2'


def _autocast_output_dtype(module, inputs, output):
    """Match nn.Linear output under autocast without casting stored weights.

    bnb Linear4bit restores the input dtype (often FP32 from LayerNorm), whereas
    torch bmm under BF16 autocast returns BF16. DeBERTa uses query.dtype for its
    masked_fill minimum, which overflows if query is FP32 but scores are BF16.
    The hook also runs on PEFT's wrapped base layer, before adding its adapter.
    """
    import torch
    if isinstance(output, torch.Tensor) and output.is_floating_point():
        device_type = output.device.type
        if torch.is_autocast_enabled(device_type):
            return output.to(dtype=torch.get_autocast_dtype(device_type))
    return output


def inspect_scope(model):
    """Inventory the original FP32 model without performing quantization."""
    import torch
    if not hasattr(model, 'encoder') or model.encoder is None:
        raise TypeError('Expected a GLiNER-like model with a loaded .encoder')
    if getattr(model, 'peft_config', None) or getattr(model, 'is_loaded_in_4bit', False):
        raise ValueError('Quantize the original base before adding/loading any adapter')
    params = dict(model.named_parameters())
    if any(p.device.type != 'cpu' for p in params.values()):
        raise ValueError('Start from original CPU weights, not a partly moved model')
    if any(p.dtype != torch.float32 for p in params.values() if p.is_floating_point()):
        raise ValueError('Load the original FP32 base; do not pre-round its weights')
    selected = []
    seen = set()
    for relative, module in model.encoder.named_modules():
        if type(module) is not torch.nn.Linear:
            continue
        if id(module.weight) in seen:
            raise ValueError('Shared linear weights need an explicit tying policy')
        seen.add(id(module.weight))
        name = 'encoder' + ('.' + relative if relative else '')
        selected.append({'module': name, 'weight': name + '.weight',
                         'shape': list(module.weight.shape), 'original_numel': module.weight.numel(),
                         'has_bias': module.bias is not None})
    if not selected:
        raise ValueError('No ordinary encoder nn.Linear modules found')
    quantized_names = {entry['weight'] for entry in selected}
    embeddings = {name + '.weight' for name, module in model.named_modules()
                  if isinstance(module, torch.nn.Embedding)}
    remaining = []
    totals = {'encoder_embeddings': 0, 'encoder_other': 0, 'task_heads': 0}
    for name, param in params.items():
        if name in quantized_names:
            continue
        group = ('encoder_embeddings' if name in embeddings and name.startswith('encoder.') else
                 'encoder_other' if name.startswith('encoder.') else 'task_heads')
        totals[group] += param.numel()
        remaining.append({'name': name, 'shape': list(param.shape), 'numel': param.numel(),
                          'dtype': str(param.dtype), 'group': group})
    total = sum(p.numel() for p in params.values())
    quantized = sum(entry['original_numel'] for entry in selected)
    assert total == quantized + sum(totals.values())
    return {'original_parameter_count': total, 'quantized_original_numel': quantized,
            'quantized_original_fraction': quantized / total,
            'quantized_linear_modules': selected, 'unquantized_parameters': remaining,
            'unquantized_numel_by_group': totals,
            'scope_sha256': hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()}


def quantize_model(model, device):
    """Convert encoder linear weights to NF4, freeze base, and return model.

    CUDA is required for a GPU request; CPU is accepted only as an explicit
    compatibility smoke backend. CPU evidence must not be called GPU validation.
    Call PEFT.get_peft_model directly afterward: GLiNER.apply_lora's dtype helper
    would incorrectly cast adapter matrices to the packed uint8 base dtype.
    Never call model.float()/half()/bfloat16() after this conversion.
    """
    import torch
    import bitsandbytes as bnb

    version = importlib.metadata.version('bitsandbytes')
    if version != BITSANDBYTES_VERSION:
        raise RuntimeError(f'Expected bitsandbytes {BITSANDBYTES_VERSION}, got {version}')
    target = torch.device(device)
    if target.type not in ('cpu', 'cuda'):
        raise ValueError('Only explicit CPU smoke or CUDA experiment is supported')
    if target.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('This NF4 configuration requires BF16 compute support')
    report = inspect_scope(model)
    for param in model.parameters():
        param.requires_grad_(False)
    for entry in report['quantized_linear_modules']:
        name = entry['module']
        old = model.get_submodule(name)
        replacement = bnb.nn.Linear4bit(old.in_features, old.out_features,
            bias=old.bias is not None, compute_dtype=torch.bfloat16,
            compress_statistics=True, quant_type='nf4', quant_storage=torch.uint8,
            device='cpu')
        # Official bnb conversion path: copy original floating-point weights,
        # then move the Linear4bit module to the target to trigger quantization.
        replacement.load_state_dict(old.state_dict(), strict=True)
        replacement.requires_grad_(False)
        replacement.register_forward_hook(_autocast_output_dtype)
        parent_name, child_name = name.rsplit('.', 1)
        setattr(model.get_submodule(parent_name), child_name, replacement)
    model.to(target)
    packed_bytes = 0
    quant_state_bytes = 0
    for entry in report['quantized_linear_modules']:
        layer = model.get_submodule(entry['module'])
        weight = layer.weight
        state = weight.quant_state
        if not isinstance(layer, bnb.nn.Linear4bit) or not weight.bnb_quantized or state is None:
            raise RuntimeError(f'Quantization did not initialize: {entry["module"]}')
        if weight.dtype != torch.uint8 or state.quant_type != 'nf4' or state.state2 is None:
            raise RuntimeError(f'Expected packed NF4 with nested statistics: {entry["module"]}')
        entry.update({'packed_shape': list(weight.shape), 'packed_dtype': str(weight.dtype),
                      'packed_weight_bytes': weight.numel() * weight.element_size(),
                      'compute_dtype': str(layer.compute_dtype), 'double_quantization': True,
                      'blocksize': state.blocksize})
        packed_bytes += entry['packed_weight_bytes']
        quant_state_bytes += sum(v.numel() * v.element_size() for v in state.as_dict(packed=True).values()
                                 if isinstance(v, torch.Tensor))
    after = dict(model.named_parameters())
    for entry in report['unquantized_parameters']:
        param = after[entry['name']]
        if str(param.dtype) != entry['dtype'] or param.numel() != entry['numel'] or param.requires_grad:
            raise RuntimeError(f'Unexpected unquantized parameter change: {entry["name"]}')
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError('Base must remain frozen before adapter injection')
    # PEFT dispatch_bnb_4bit requires this flag for its Linear4bit LoRA wrapper.
    model.is_loaded_in_4bit = True
    model.is_loaded_in_8bit = False
    report.update({'method': 'encoder-linear NF4 + double quantization; FP32 embeddings/norms/heads',
                   'bitsandbytes_version': version, 'torch_version': torch.__version__,
                   'device': str(target), 'compute_dtype': 'torch.bfloat16',
                   'packed_weight_bytes': packed_bytes, 'quant_state_tensor_bytes': quant_state_bytes,
                   'base_trainable_parameters_before_adapters': 0,
                   'gradient_checkpointing': False, 'quantization_completed': True,
                   'autocast_output_policy': 'Linear4bit forward outputs use the active device autocast dtype; disabled-autocast output behavior is unchanged. Hook is installed before PEFT wrapping.',
                   'linear_bias_policy': 'Initial FP32, frozen; bitsandbytes casts bias storage to BF16 on forward. Unquantized dtype list records the post-load state.',
                   'cuda_quantization_executed': target.type == 'cuda',
                   'gpu_training_validated': False,
                   'note': 'Packed-weight size is not peak VRAM; embeddings and task heads are not 4-bit.'})
    model._crm_quantization_report = report
    return model


if __name__ == '__main__':
    import argparse
    from gliner2 import AutoExtractor
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--inventory-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = AutoExtractor.from_pretrained(str(args.base.resolve()), map_location='cpu').float()
    result = inspect_scope(model) if args.inventory_only else quantize_model(model, args.device)._crm_quantization_report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('original_parameter_count', 'quantized_original_numel', 'unquantized_numel_by_group')}))
