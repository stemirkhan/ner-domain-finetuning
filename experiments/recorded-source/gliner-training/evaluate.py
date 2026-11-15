"""Dev selection, frozen test evaluation and a train-only runtime smoke check.

Candidate caching is an explicit common decoder: request all returned proposals
(threshold 0, overlaps allowed), retain checkpoint abstention, then threshold and
resolve overlaps globally. It is not a cache of already-flat predictions.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

from metrics import DECODER, decode, evaluate, mask_text, selection_key, validate_spans

THRESHOLDS = [i / 20 for i in range(1, 20)]


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def path_at(root, value):
    p = Path(value)
    return p if p.is_absolute() else Path(root) / p


def relative_path(root, path):
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(Path(path).resolve())


def directory_hashes(path):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(p for p in path.rglob('*') if p.is_file() and not any(part.startswith('.') for part in p.relative_to(path).parts))
    if not any(p.suffix == '.safetensors' for p in files):
        raise ValueError(f'No safetensors in model/adapter directory: {path}')
    return {str(p.relative_to(path)): sha(p) for p in files}


def runtime_hashes():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def dependency_versions():
    from importlib.metadata import version, PackageNotFoundError
    import platform
    result = {'python': platform.python_version()}
    for name in ('torch', 'gliner2', 'transformers', 'peft', 'tokenizers', 'numpy'):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def data_hashes(root):
    data = Path(root) / 'data/corpus-v1'
    return {**{split: sha(data / f'{split}.jsonl') for split in ('train', 'dev', 'test')},
            'schema': sha(data / 'model-label-schema.json')}


def save_jsonl(path, rows):
    with Path(path).open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def compact_metrics(metrics):
    return {k: v for k, v in metrics.items() if k != 'document_details'}


def synchronize(device):
    if str(device).startswith('cuda'):
        import torch
        torch.cuda.synchronize(device)


class Predictor:
    """Only text + fixed label descriptions enter extract_entities."""
    def __init__(self, root, model_path, device, precision, adapter_path=None):
        from common import load_json, load_model
        import torch
        self.device, self.precision = device, precision
        if precision == 'bf16' and (not str(device).startswith('cuda') or not torch.cuda.is_bf16_supported()):
            raise ValueError('BF16 evaluation requires a supported CUDA device')
        self.schema = load_json(Path(root) / 'data/corpus-v1/model-label-schema.json')
        self.reverse = {name: label for label, name in self.schema['names'].items()}
        self.descriptions = self.schema['descriptions']
        if set(self.reverse) != set(self.descriptions):
            raise ValueError('Schema descriptions and labels differ')
        started = time.perf_counter()
        self.model = load_model(path_at(root, model_path), device,
                                adapter_path=path_at(root, adapter_path) if adapter_path else None)
        settings = getattr(self.model, 'boundary_settings', None)
        if settings is None or settings.adaptive_threshold:
            raise ValueError('Candidate cache requires boundary architecture with adaptive_threshold=False')
        synchronize(device)
        self.load_seconds = time.perf_counter() - started
        self.adapter_merged = bool(adapter_path)
        self.settings = {
            'adaptive_threshold': settings.adaptive_threshold,
            'abstention_threshold': getattr(settings, 'abstention_threshold', None),
            'enable_abstention': getattr(settings, 'enable_abstention', None),
            'candidate_budget': getattr(settings, 'candidate_budget', None),
            'pair_temperature': getattr(settings, 'pair_temperature', None),
        }

    def candidates(self, texts, batch_size=1):
        import torch
        autocast = torch.autocast('cuda', dtype=torch.bfloat16) if self.precision == 'bf16' else nullcontext()
        with torch.inference_mode(), autocast:
            results = self.model.batch_extract_entities(
                texts, self.descriptions, batch_size=batch_size, threshold=0.0,
                include_spans=True, include_confidence=True, max_len=None, overlap_policy='allow')
        if len(results) != len(texts):
            raise ValueError('Extractor changed batch length')
        all_candidates = []
        for text, result in zip(texts, results):
            candidates = []
            for name, values in result['entities'].items():
                if name not in self.reverse:
                    raise ValueError(f'Unexpected model label: {name}')
                for value in values:
                    candidates.append({'label': self.reverse[name], 'start': int(value['start']),
                                       'end': int(value['end']), 'text': value['text'],
                                       'score': float(value['confidence'])})
            validate_spans(text, candidates, self.schema['names'], scored=True)
            all_candidates.append(candidates)
        return all_candidates

    def predict(self, texts, threshold, batch_size=1):
        return [decode(candidates, threshold) for candidates in self.candidates(texts, batch_size)]

    def close(self):
        del self.model
        gc.collect()
        if str(self.device).startswith('cuda'):
            import torch
            torch.cuda.empty_cache()


def collect_candidates(predictor, rows, out, batch_size):
    """Stream actual candidates to disk, retaining partial evidence on failure."""
    records = []
    started = time.perf_counter()
    with Path(out).open('x') as f:
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset+batch_size]
            values = predictor.candidates([row['text'] for row in batch], batch_size)
            for row, candidates in zip(batch, values):
                item = {'id': row['id'], 'text': row['text'], 'candidates': candidates}
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                records.append(item)
            f.flush()
    synchronize(predictor.device)
    return records, time.perf_counter()-started


def decoded_rows(raw, threshold):
    return [{'id': row['id'], 'text': row['text'], 'predictions': decode(row['candidates'], threshold)} for row in raw]


def validate_rows(rows, split, expected):
    if len(rows) != expected or {row['split'] for row in rows} != {split}:
        raise ValueError(f'Expected all {expected} rows of {split}, not a subset')
    if len({r['id'] for r in rows}) != expected:
        raise ValueError('Duplicate data ID')


def checkpoint_record(root, candidate):
    item = dict(candidate)
    item.setdefault('adapter_path', None)
    item.setdefault('run_manifest', None)
    if not isinstance(item.get('epoch'), int) or item['epoch'] < 0:
        raise ValueError('Candidate epoch must be a nonnegative integer')
    if not item.get('name') or not item.get('model_path'):
        raise ValueError('Candidate requires name and model_path')
    item['model_files'] = directory_hashes(path_at(root, item['model_path']))
    item['adapter_files'] = directory_hashes(path_at(root, item['adapter_path'])) if item['adapter_path'] else None
    if item['run_manifest']:
        item['run_manifest_sha256'] = sha(path_at(root, item['run_manifest']))
    return item


def check_checkpoint(root, selected):
    if directory_hashes(path_at(root, selected['model_path'])) != selected['model_files']:
        raise ValueError('Selected model files changed after dev selection')
    if selected.get('adapter_path') and directory_hashes(path_at(root, selected['adapter_path'])) != selected['adapter_files']:
        raise ValueError('Selected adapter files changed after dev selection')
    if selected.get('run_manifest') and sha(path_at(root, selected['run_manifest'])) != selected['run_manifest_sha256']:
        raise ValueError('Training run manifest changed after dev selection')


def dev_select(args):
    from common import load_json, write_json, verify_data, seed_all
    root = Path(args.root); out = path_at(root, args.output)
    verify_data(root); seed_all(20261114)
    rows = read_jsonl(root/'data/corpus-v1/dev.jsonl'); validate_rows(rows, 'dev', 500)
    candidates = load_json(path_at(root, args.candidates))
    if not isinstance(candidates, list) or not candidates:
        raise ValueError('Candidates JSON must be a nonempty list')
    if len({c['name'] for c in candidates}) != len(candidates):
        raise ValueError('Duplicate checkpoint candidate names')
    if args.variant == 'base' and (len(candidates) != 1 or candidates[0]['epoch'] != 0 or candidates[0].get('adapter_path')):
        raise ValueError('Base must have one unadapted epoch=0 candidate')
    if args.variant != 'base' and any(c['epoch'] < 1 or not c.get('run_manifest') for c in candidates):
        raise ValueError('Trained candidates require epochs >=1 and a main training run manifest')
    if args.variant == 'base':
        validate_main_run(root, {'variant': 'base', 'selected': candidates[0]})
    if args.variant != 'base':
        manifests = {c['run_manifest'] for c in candidates}
        if len(manifests) != 1:
            raise ValueError('One main training run per dev selection is required')
        run = load_json(path_at(root, next(iter(manifests))))
        expected_epochs = set(range(1, int(run.get('config', {}).get('num_epochs', 0))+1))
        if not expected_epochs or {c['epoch'] for c in candidates} != expected_epochs or len(candidates) != len(expected_epochs):
            raise ValueError('Dev selection requires every epoch checkpoint exactly once')
        for candidate in candidates:
            validate_main_run(root, {'variant': args.variant, 'selected': candidate})
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'invocation.json', vars(args))
    labels = list(load_json(root/'data/corpus-v1/model-label-schema.json')['names'])
    results = []
    for index, candidate in enumerate(candidates):
        checkpoint = checkpoint_record(root, candidate)
        checkpoint_out = out/f'checkpoint-{index:02d}'
        checkpoint_out.mkdir()
        predictor = Predictor(root, checkpoint['model_path'], args.device, args.precision, checkpoint['adapter_path'])
        try:
            raw, seconds = collect_candidates(predictor, rows, checkpoint_out/'raw-candidates.jsonl', args.batch_size)
            write_json(checkpoint_out/'runtime.json', {'load_seconds': predictor.load_seconds,
                       'inference_seconds': seconds, 'boundary_settings': predictor.settings,
                       'adapter_merged': predictor.adapter_merged, 'precision': args.precision, 'device': args.device})
        finally:
            predictor.close()
        for threshold in THRESHOLDS:
            pred = decoded_rows(raw, threshold)
            m = evaluate(rows, pred, labels)
            key = f'{threshold:.2f}'
            save_jsonl(checkpoint_out/f'predictions-{key}.jsonl', pred)
            write_json(checkpoint_out/f'metrics-{key}.json', m)
            result = {**checkpoint, 'threshold': threshold, 'metrics': compact_metrics(m)}
            results.append(result)
            write_json(out/'candidates.json', results)
        print(json.dumps({'checkpoint': candidate['name'], 'dev_documents': len(rows), 'thresholds': len(THRESHOLDS)}), flush=True)
    selected = max(results, key=selection_key)
    selection = {'version': 1, 'status': 'complete', 'variant': args.variant, 'split': 'dev', 'documents': len(rows),
                 'created_utc': datetime.now(timezone.utc).isoformat(), 'precision': args.precision,
                 'batch_size': args.batch_size, 'decoder': DECODER, 'threshold_grid': THRESHOLDS,
                 'criterion': 'max strict micro-F1; fewer masked public targets; earlier epoch; higher threshold',
                 'data_hashes': {k:v for k,v in data_hashes(root).items() if k in ('dev','schema')},
                 'runtime_hashes': runtime_hashes(), 'dependency_versions': dependency_versions(), 'selected': selected,
                 'candidate_count': len(results), 'checkpoint_count': len(candidates),
                 'candidates_sha256': sha(out/'candidates.json')}
    write_json(out/'selection.json', selection)
    print(json.dumps({'variant': args.variant, 'checkpoint': selected['name'], 'threshold': selected['threshold'], 'micro': selected['metrics']['micro']}), flush=True)


def validate_main_run(root, selection):
    from common import load_json, verify_data, BASE_SHA256
    selected = selection['selected']
    if selection['variant'] == 'base':
        if selected['epoch'] != 0 or selected.get('adapter_path'):
            raise ValueError('Base selection must use unadapted epoch=0')
        if sha(path_at(root, selected['model_path'])/'model.safetensors') != BASE_SHA256:
            raise ValueError('Base must use the pinned original model weights')
        if directory_hashes(path_at(root, selected['model_path'])) != directory_hashes(Path(root)/'model'):
            raise ValueError('Base config/tokenizer must match the original bundle model')
        return
    if selection['variant'] == 'full' and selected.get('adapter_path'):
        raise ValueError('Full fine-tuning must use a full checkpoint, not an adapter')
    if selection['variant'] == 'lora':
        if not selected.get('adapter_path') or sha(path_at(root, selected['model_path'])/'model.safetensors') != BASE_SHA256:
            raise ValueError('LoRA must use an adapter on the pinned original base')
        if directory_hashes(path_at(root, selected['model_path'])) != directory_hashes(Path(root)/'model'):
            raise ValueError('LoRA base config/tokenizer must match the original bundle model')
    if not selected.get('run_manifest'):
        raise ValueError('Training run manifest missing')
    run = load_json(path_at(root, selected['run_manifest']))
    if (run.get('run_kind') != 'main' or run.get('status') != 'completed'
            or run.get('mode') != selection['variant'] or run.get('training_examples') != 5000):
        raise ValueError('Final test requires a completed main run on all 5000 train documents, not smoke/pilot')
    if run.get('data_verification') != verify_data(root):
        raise ValueError('Main training used different frozen data/schema')
    manifest_path = path_at(root, selected['run_manifest'])
    selected_dir = path_at(root, selected.get('adapter_path') or selected['model_path']).resolve()
    matching = [cp for cp in run.get('checkpoints', [])
                if path_at(manifest_path.parent, cp['path']).resolve() == selected_dir
                and cp.get('epoch') == selected['epoch']]
    if len(matching) != 1:
        raise ValueError('Selected checkpoint does not belong to its declared main training run')


def freeze(args):
    from common import load_json, write_json, verify_data
    root = Path(args.root); out = path_at(root, args.output)
    verify_data(root)
    if out.exists():
        raise FileExistsError(out)
    variants, precisions, batch_sizes = {}, set(), set()
    hashes = data_hashes(root)
    for variant in ('base', 'full', 'lora'):
        path = path_at(root, getattr(args, variant+'_selection'))
        selection = load_json(path)
        if selection.get('status') != 'complete' or selection.get('variant') != variant or selection.get('split') != 'dev' or selection.get('documents') != 500:
            raise ValueError('Only completed dev selection on all 500 documents can be frozen')
        if selection['decoder'] != DECODER or selection['threshold_grid'] != THRESHOLDS or selection['runtime_hashes'] != runtime_hashes() or selection.get('dependency_versions') != dependency_versions():
            raise ValueError('Decoder, threshold grid, runtime or dependencies changed since dev selection')
        if selection['data_hashes'] != {k: hashes[k] for k in ('dev', 'schema')}:
            raise ValueError('Dev data/schema changed since selection')
        if not path.parent.joinpath('candidates.json').exists() or sha(path.parent/'candidates.json') != selection['candidates_sha256']:
            raise ValueError('Missing/changed complete candidate evidence')
        all_candidates = load_json(path.parent/'candidates.json')
        expected_count = selection['checkpoint_count'] * len(THRESHOLDS)
        if len(all_candidates) != expected_count or selection['candidate_count'] != expected_count:
            raise ValueError('Incomplete dev threshold/checkpoint sweep')
        if max(all_candidates, key=selection_key) != selection['selected']:
            raise ValueError('Selection does not follow the fixed dev criterion')
        if variant != 'base':
            run = load_json(path_at(root, selection['selected']['run_manifest']))
            expected_epochs = set(range(1, int(run.get('config', {}).get('num_epochs', 0))+1))
            if {c['epoch'] for c in all_candidates} != expected_epochs:
                raise ValueError('An epoch was omitted from dev selection')
        for checkpoint_name in {c['name'] for c in all_candidates}:
            thresholds = [c['threshold'] for c in all_candidates if c['name'] == checkpoint_name]
            if sorted(thresholds) != THRESHOLDS:
                raise ValueError('A checkpoint has missing or repeated thresholds')
        check_checkpoint(root, selection['selected']); validate_main_run(root, selection)
        variants[variant] = {'selection_path': relative_path(root, path), 'selection_sha256': sha(path)}
        precisions.add(selection['precision']); batch_sizes.add(selection['batch_size'])
    if len(precisions) != 1 or len(batch_sizes) != 1:
        raise ValueError('All variants must use the same inference precision and dev batch size')
    write_json(out, {'version': 1, 'purpose': 'crm_final_test', 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'precision': precisions.pop(), 'benchmark_batch_size': batch_sizes.pop(), 'decoder': DECODER, 'data_hashes': hashes,
                    'runtime_hashes': runtime_hashes(), 'dependency_versions': dependency_versions(), 'variants': variants})
    print(f'Frozen final evaluation: {out}', flush=True)


def load_frozen(root, freeze_path, variant):
    from common import load_json, verify_data
    verify_data(root)
    freeze_path = path_at(root, freeze_path)
    frozen = load_json(freeze_path)
    if frozen.get('purpose') != 'crm_final_test' or set(frozen.get('variants', {})) != {'base', 'full', 'lora'}:
        raise ValueError('A final freeze containing base/full/lora is required')
    if frozen['data_hashes'] != data_hashes(root) or frozen['runtime_hashes'] != runtime_hashes() or frozen['decoder'] != DECODER or frozen.get('dependency_versions') != dependency_versions():
        raise ValueError('Frozen data, runtime, dependencies or decoder changed')
    item = frozen['variants'][variant]
    path = path_at(root, item['selection_path'])
    if sha(path) != item['selection_sha256']:
        raise ValueError('Selection changed after freeze')
    selection = load_json(path)
    if selection['precision'] != frozen['precision'] or selection['decoder'] != DECODER:
        raise ValueError('Frozen inference policy mismatch')
    check_checkpoint(root, selection['selected']); validate_main_run(root, selection)
    return frozen, selection


def test(args):
    from common import load_json, write_json, seed_all
    root = Path(args.root); out = path_at(root, args.output)
    args.freeze = str(path_at(root, args.freeze))
    frozen, selection = load_frozen(root, args.freeze, args.variant)
    rows = read_jsonl(root/'data/corpus-v1/test.jsonl'); validate_rows(rows, 'test', 1000)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'invocation.json', vars(args))
    write_json(out/'freeze-reference.json', {'path': str(args.freeze), 'sha256': sha(args.freeze)})
    seed_all(20261114)
    selected = selection['selected']
    predictor = Predictor(root, selected['model_path'], args.device, frozen['precision'], selected.get('adapter_path'))
    try:
        raw, seconds = collect_candidates(predictor, rows, out/'raw-candidates.jsonl', args.batch_size)
        write_json(out/'runtime.json', {'load_seconds': predictor.load_seconds, 'inference_seconds': seconds,
                   'boundary_settings': predictor.settings, 'adapter_merged': predictor.adapter_merged,
                   'device': args.device, 'precision': frozen['precision'], 'batch_size': args.batch_size})
    finally:
        predictor.close()
    labels = list(load_json(root/'data/corpus-v1/model-label-schema.json')['names'])
    for label, threshold in (('selected', selected['threshold']), ('common-0.50', .5)):
        predictions = decoded_rows(raw, threshold)
        save_jsonl(out/f'predictions-{label}.jsonl', predictions)
        m = evaluate(rows, predictions, labels)
        write_json(out/f'metrics-{label}.json', m)
        save_jsonl(out/f'masked-{label}.jsonl', [{'id': p['id'], 'text': p['text'],
                   'masked_text': mask_text(p['text'], p['predictions'])} for p in predictions])
    write_json(out/'complete.json', {'status': 'complete', 'variant': args.variant, 'documents': len(rows),
               'selected_threshold': selected['threshold'], 'freeze_sha256': sha(args.freeze)})
    print(json.dumps({'variant': args.variant, 'test_documents': len(rows), 'threshold': selected['threshold']}), flush=True)


def smoke(args):
    from common import write_json, verify_data, seed_all
    root = Path(args.root); out = path_at(root, args.output)
    verify_data(root); seed_all(20261114)
    if args.max_samples < 1 or args.max_samples > 32:
        raise ValueError('Smoke is restricted to 1-32 train samples')
    rows = read_jsonl(root/'data/corpus-v1/train.jsonl')[:args.max_samples]
    if any(row['split'] != 'train' for row in rows):
        raise ValueError('Smoke must use train only')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'invocation.json', vars(args))
    predictor = Predictor(root, args.model, args.device, args.precision, args.adapter)
    try:
        raw, seconds = collect_candidates(predictor, rows, out/'raw-candidates.jsonl', 1)
        pred = decoded_rows(raw, .5); save_jsonl(out/'predictions.jsonl', pred)
        write_json(out/'complete.json', {'status': 'complete', 'scope': 'train-only inference smoke',
                   'documents': len(rows), 'load_seconds': predictor.load_seconds, 'inference_seconds': seconds,
                   'device': args.device, 'precision': args.precision, 'boundary_settings': predictor.settings})
    finally:
        predictor.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for command in ('dev-select', 'freeze', 'test', 'smoke'):
        q = sub.add_parser(command); q.add_argument('--root', required=True); q.add_argument('--output', required=True)
        if command == 'freeze':
            for variant in ('base', 'full', 'lora'):
                q.add_argument('--'+variant+'-selection', required=True)
            continue
        q.add_argument('--device', default='cuda')
        if command != 'test':
            q.add_argument('--precision', choices=('bf16','fp32'), default='bf16')
        if command == 'smoke':
            q.add_argument('--model', default='model'); q.add_argument('--adapter'); q.add_argument('--max-samples', type=int, default=2)
        else:
            q.add_argument('--variant', choices=('base','full','lora'), required=True)
            q.add_argument('--batch-size', type=int, default=2)
            if command == 'test': q.add_argument('--freeze', required=True)
            else: q.add_argument('--candidates', required=True)
    return p


def main():
    args = parser().parse_args()
    if hasattr(args, 'batch_size') and args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    if path_at(args.root, args.output).exists():
        raise FileExistsError(path_at(args.root, args.output))
    try:
        {'dev-select': dev_select, 'freeze': freeze, 'test': test, 'smoke': smoke}[args.command](args)
    except Exception as exc:
        out = path_at(args.root, args.output)
        if out.is_dir() and not (out/'failure.json').exists():
            from common import write_json
            write_json(out/'failure.json', {'type': type(exc).__name__, 'message': str(exc),
                                          'utc': datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == '__main__':
    main()
