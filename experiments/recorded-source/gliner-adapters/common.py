"""Portable helpers shared by the rental runtime. No model load at import."""
import hashlib
import json
import os
from pathlib import Path
import random

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
ROOT = Path(__file__).resolve().parents[1]
BASE_REVISION = 'cf5593a5d45e3bbf204b9df621b13b1c0cf25ee3'
BASE_SHA256 = 'c1ff4ec0bc00031c15530b8f3c33d3677f27949e6a0cb52e1247a6224b6c5395'


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_data(root):
    root = Path(root)
    data = root / 'data/corpus-v1'
    frozen = load_json(root / 'data-freeze.json')
    if sha256(data / 'manifest.json') != frozen['data_manifest_sha256']:
        raise ValueError('Frozen data manifest changed')
    checked = {}
    for relative, info in load_json(data / 'manifest.json')['files'].items():
        path = (data / relative).resolve()
        if not path.is_relative_to(data.resolve()):
            raise ValueError('Unsafe path in data manifest')
        if sha256(path) != info['sha256'] or path.stat().st_size != info['bytes']:
            raise ValueError(f'Data file changed: {relative}')
        checked[relative] = info['sha256']
    for split, info in frozen['splits'].items():
        if checked[f'{split}.jsonl'] != info['raw_sha256'] or checked[f'gliner/{split}.jsonl'] != info['gliner_sha256']:
            raise ValueError(f'Frozen split changed: {split}')
    return {'status': 'pass', 'files_checked': len(checked), 'data_manifest_sha256': frozen['data_manifest_sha256'],
            'splits': frozen['splits']}


def seed_all(seed):
    random.seed(seed)
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_processor(model):
    sampling = model.processor.sampling_config
    for key in ('remove_entities_prob', 'remove_entity_prob', 'synthetic_entity_label_prob'):
        setattr(sampling, key, 0.0)
    sampling.shuffle_entities = False
    return model


def load_model(model_path, device='cuda', adapter_path=None, variant=None):
    import torch
    from gliner2 import AutoExtractor
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. CPU checks require an explicit --device cpu.')
    model = AutoExtractor.from_pretrained(str(Path(model_path).resolve()), map_location='cpu').float()
    if variant == 'qlora':
        from quantization import quantize_model
        model = quantize_model(model, device)
    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(Path(adapter_path).resolve()))
        if variant != 'qlora':
            model = model.merge_and_unload()
    configure_processor(model)
    return model.to(device).eval()
