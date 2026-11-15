"""Portable educational GLiNER2.5 Full/LoRA recipe, separate from recorded experiments."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random

BASE_ID = 'fastino/gliner2.5-multi-v1'
BASE_REVISION = 'cf5593a5d45e3bbf204b9df621b13b1c0cf25ee3'
BASE_WEIGHTS_SHA256 = 'c1ff4ec0bc00031c15530b8f3c33d3677f27949e6a0cb52e1247a6224b6c5395'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def read_schema(path):
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    schema = {k: raw[k] for k in ('names', 'descriptions')}
    names, descriptions = schema['names'], schema['descriptions']
    if not names or len(set(names.values())) != len(names) or set(names.values()) != set(descriptions):
        raise ValueError('Each label needs one unique model name and one description')
    if not all(isinstance(x, str) and x.strip() for x in [*names, *names.values(), *descriptions.values()]):
        raise ValueError('Schema labels, names and descriptions must be nonempty strings')
    return schema


def validate_spans(text, spans, labels, scored=False):
    if not isinstance(spans, list): raise ValueError('spans must be a list, including [] for negative examples')
    seen = set()
    for s in spans:
        a, b = s['start'], s['end']
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(text):
            raise ValueError('Invalid half-open character coordinates')
        if s['label'] not in labels or s['text'] != text[a:b]: raise ValueError('Unknown label or incorrect source substring')
        key = s['label'], a, b
        if key in seen: raise ValueError('Duplicate span')
        seen.add(key)
        if scored and (not math.isfinite(float(s['score'])) or not 0 <= s['score'] <= 1):
            raise ValueError('Invalid prediction confidence')
    if not scored:
        ordered = sorted(spans, key=lambda x: (x['start'], x['end']))
        if any(a['end'] > b['start'] for a, b in zip(ordered, ordered[1:])):
            raise ValueError('This flat NER recipe does not accept overlapping gold spans')


def read_rows(path, schema):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows: raise ValueError('Empty corpus')
    ids = set()
    for row in rows:
        if not isinstance(row.get('id'), str) or not row['id'] or row['id'] in ids: raise ValueError('Missing/duplicate document ID')
        if not isinstance(row.get('text'), str) or not row['text']: raise ValueError('Missing document text')
        if 'spans' not in row: raise ValueError('Missing spans is not a negative example; use [] explicitly')
        validate_spans(row['text'], row['spans'], schema['names']); ids.add(row['id'])
    return rows


def to_example(row, schema):
    """Convert only text, labels and explicit offsets; no family/private metadata."""
    validate_spans(row['text'], row['spans'], schema['names'])
    entities = {name: [] for name in schema['names'].values()}
    for span in row['spans']:
        entities[schema['names'][span['label']]].append({k: span[k] for k in ('text', 'start', 'end')})
    return {'input': row['text'], 'output': {'entities': entities, 'entity_descriptions': schema['descriptions']}}


def decode(candidates, threshold):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1: raise ValueError('Threshold must be finite in [0,1]')
    ranked = sorted((s for s in candidates if s['score'] >= threshold),
                    key=lambda s: (-s['score'], -(s['end']-s['start']), s['start'], s['end'], s['label']))
    kept = []
    for span in ranked:
        if not any(span['start'] < other['end'] and other['start'] < span['end'] for other in kept): kept.append(span)
    return sorted(kept, key=lambda s: (s['start'], s['end'], s['label']))


def mask(text, spans):
    last = len(text)
    for s in sorted(spans, key=lambda s: s['start'], reverse=True):
        if s['end'] > last or text[s['start']:s['end']] != s['text']: raise ValueError('Mask requires valid non-overlapping spans')
        text = text[:s['start']]+'['+s['label']+']'+text[s['end']:]; last = s['start']
    return text


def exact_counts(rows, predictions):
    tp = fp = fn = 0
    for row, pred in zip(rows, predictions, strict=True):
        gold = {(s['label'], s['start'], s['end']) for s in row['spans']}
        guessed = {(s['label'], s['start'], s['end']) for s in pred}
        tp += len(gold & guessed); fp += len(guessed-gold); fn += len(gold-guessed)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def check_versions():
    from importlib.metadata import version
    for name, expected in [('gliner2', '2.1.0'), ('transformers', '5.22.1'), ('peft', '0.21.2')]:
        if version(name) != expected: raise RuntimeError(f'Tested API requires {name}=={expected}')


def base_directory(model=None):
    if model is None:
        from huggingface_hub import snapshot_download
        model = snapshot_download(BASE_ID, revision=BASE_REVISION, token=False,
                                  allow_patterns=['*.json', '*.model', '*.txt', '*.safetensors', 'encoder_config/*'])
    p = Path(model)
    if sha(p/'model.safetensors') != BASE_WEIGHTS_SHA256: raise ValueError('Base weights differ from pinned checkpoint')
    return p


def load_base(model=None):
    from gliner2 import AutoExtractor
    m = AutoExtractor.from_pretrained(str(base_directory(model).resolve()), map_location='cpu').float()
    if getattr(m, 'architecture', None) != 'boundary': raise ValueError('Expected GLiNER2.5 boundary extractor')
    sampling = m.processor.sampling_config
    for name in ('remove_entities_prob', 'remove_entity_prob', 'synthetic_entity_label_prob'): setattr(sampling, name, 0.0)
    sampling.shuffle_entities = False
    return m


def trainer_class():
    import torch
    from torch.utils.data import DataLoader
    from gliner2.training.trainer import ExtractorTrainer

    class FixedSchemaTrainer(ExtractorTrainer):
        def __init__(self, model, config, schema, device, cap):
            self.requested_device, self.schema, self.cap = device, schema, cap
            self.processed_examples, self.checkpoints = 0, []
            super().__init__(model, config)

        def _setup_device(self):
            if self.requested_device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but not available')
            self.device = torch.device(self.requested_device); self.is_distributed = False
            self.model.to(self.device).float()

        def collate(self, records):
            batch = self.processor.collate_fn_inference(records, architecture='boundary', max_len=None,
                        error_policy='raise', build_targets=True, max_gold_per_query=64, on_capacity_exceeded='raise')
            if len(batch) != len(records) or batch.input_ids.shape[-1] > self.cap:
                raise ValueError('Dropped record or over-length input; truncation is forbidden')
            for i, (text, output) in enumerate(records):
                expected = {(label, s['start'], s['end']) for label, values in output['entities'].items() for s in values}
                actual = set(); order = []
                for qi, query in enumerate(batch.query_layouts[i].queries):
                    order.append(query.role_name)
                    pairs, valid = batch.targets.mention_pairs[i, qi], batch.targets.mention_mask[i, qi]
                    for a, b in pairs[valid].tolist(): actual.add((query.role_name, batch.start_mappings[i][a], batch.end_mappings[i][b-1]))
                if actual != expected or order != list(self.schema['names'].values()): raise ValueError('Gold target roundtrip or schema order failed')
            return batch

        def _create_dataloader(self, dataset, batch_size, shuffle=True, is_training=True):
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
                              num_workers=0, collate_fn=self.collate, pin_memory=self.device.type == 'cuda')

        def _backward_one(self, batch, *args, **kwargs):
            try: result = super()._backward_one(batch, *args, **kwargs)
            except torch.cuda.OutOfMemoryError as error: raise RuntimeError('OOM: abort without dropping examples') from error
            self.processed_examples += len(batch)
            return result

        def _save_checkpoint(self, name):
            # Keep epoch checkpoints; final is saved only for a bounded smoke run.
            if name == 'final' and self.config.max_steps <= 0: return
            directory = self.output_dir/name
            if directory.exists(): raise FileExistsError(directory)
            super()._save_checkpoint(name)
            weights = {p.name: sha(p) for p in directory.glob('*.safetensors')}
            save(directory/'recipe.json', {'mode': 'lora' if self.config.use_lora else 'full', 'base_id': BASE_ID,
                    'base_revision': BASE_REVISION, 'schema': self.schema, 'weights_sha256': weights,
                    'epoch': self.epoch+1, 'max_encoded_tokens': self.cap, 'exact_optimizer_resume': False})
            self.checkpoints.append(name)
    return FixedSchemaTrainer


def train(args):
    check_versions()
    import torch
    from gliner2.training.trainer import TrainingConfig
    schema = read_schema(args.schema); rows = read_rows(args.data, schema)
    cfg = json.loads(Path(args.config).read_text()); mode = cfg['mode']
    if mode not in ('full', 'lora'): raise ValueError('Supported modes: full/lora')
    if args.device == 'cpu' and cfg['precision'] != 'fp32': raise ValueError('Use a separate fp32 config for an explicit CPU smoke test')
    if cfg['precision'] not in ('fp32', 'bf16'): raise ValueError('Use fp32 or bf16 precision')
    if cfg['precision'] == 'bf16' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()): raise RuntimeError('BF16 requires supported CUDA')
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.cpu_threads); random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg['seed'])
    state = {'status': 'running', 'recipe': 'educational-full-lora-v1', 'mode': mode, 'config': cfg,
             'train_sha256': sha(args.data), 'schema': schema, 'training_ids': [r['id'] for r in rows],
             'training_text_hashes': [hashlib.sha256(r['text'].encode()).hexdigest() for r in rows],
             'smoke_steps': args.smoke_steps, 'test_calls': 0,
             'experiment_relation': 'Portable educational recipe; not a claim of an identical experiment replay'}
    save(output/'run.json', state)
    try:
        model = load_base(args.model)
        if mode == 'full':
            for p in model.parameters(): p.requires_grad = True
        config = TrainingConfig(output_dir=str(output), num_epochs=cfg['num_epochs'],
            max_steps=args.smoke_steps or -1, batch_size=cfg['batch_size'], gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
            encoder_lr=cfg['encoder_lr'], task_lr=cfg['task_lr'], scheduler_type='linear',
            warmup_ratio=cfg['warmup_ratio'], weight_decay=cfg['weight_decay'], max_grad_norm=1.0,
            fp16=False, bf16=cfg['precision']=='bf16', seed=cfg['seed'], deterministic=True,
            use_lora=mode=='lora', lora_r=16, lora_alpha=32, lora_dropout=.05,
            lora_target_modules=['encoder.query', 'encoder.value'], save_adapter_only=True,
            eval_strategy='no' if args.smoke_steps else 'epoch', early_stopping=False, save_best=False,
            save_total_limit=0, num_workers=0, max_len=None, strict_training=True, skip_step_errors=False,
            allow_invalid_samples=False, ignore_nonfinite_losses=False, report_to_wandb=False,
            compile_model=False, gradient_checkpointing=False, fused_optimizer=args.device=='cuda',
            allow_tf32=False, float32_matmul_precision='highest', logging_steps=10)
        trainer = trainer_class()(model, config, schema, args.device, cfg['max_encoded_tokens'])
        learned = [(name, p) for name, p in trainer.model.named_parameters() if p.requires_grad]
        if mode == 'lora' and any('lora_A' not in n and 'lora_B' not in n for n, _ in learned): raise ValueError('Unexpected unfrozen weights')
        before = {name: p.detach().flatten()[:64].cpu().clone() for name, p in learned}
        trainer.train(train_data=[to_example(row, schema) for row in rows])
        changed = any(not torch.equal(before[n], p.detach().flatten()[:64].cpu()) for n, p in learned)
        if not changed: raise RuntimeError('No sampled trainable parameter changed')
        if not args.smoke_steps and trainer.processed_examples != len(rows)*cfg['num_epochs']:
            raise RuntimeError('Training did not consume all examples in all epochs')
        if args.smoke_steps and trainer.global_step != args.smoke_steps: raise RuntimeError('Incomplete smoke run')
        state.update(status='complete', checkpoints=trainer.checkpoints, trainable_parameters=sum(p.numel() for _, p in learned),
                     examples_processed=trainer.processed_examples, optimizer_steps=trainer.global_step, sampled_parameters_changed=changed)
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__); save(output/'run.json', state); raise
    save(output/'run.json', state); print(json.dumps({k: state[k] for k in ('status','checkpoints','trainable_parameters','optimizer_steps')}))


def read_checkpoint(directory):
    p = Path(directory); recipe = json.loads((p/'recipe.json').read_text())
    if recipe.get('mode') not in ('full', 'lora') or recipe.get('base_id') != BASE_ID or recipe.get('base_revision') != BASE_REVISION:
        raise ValueError('Foreign/missing checkpoint recipe')
    if not recipe.get('weights_sha256'): raise ValueError('Checkpoint has no saved weights')
    for name, digest in recipe['weights_sha256'].items():
        if Path(name).name != name or sha(p/name) != digest: raise ValueError('Changed/unsafe checkpoint weight')
    return recipe


def load_checkpoint(directory, device='cpu', model=None):
    check_versions()
    import torch
    from gliner2 import AutoExtractor
    recipe = read_checkpoint(directory)
    if device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    if recipe['mode'] == 'lora':
        from peft import PeftModel
        m = PeftModel.from_pretrained(load_base(model), str(directory)).merge_and_unload()
    else: m = AutoExtractor.from_pretrained(str(Path(directory).resolve()), map_location='cpu').float()
    return m.to(device).eval(), recipe


def candidates(model, text, schema, max_encoded_tokens=1024):
    import torch
    reverse = {name: label for label, name in schema['names'].items()}
    empty = {'entities': {name: [] for name in schema['names'].values()}, 'entity_descriptions': schema['descriptions']}
    batch = model.processor.collate_fn_inference([(text, empty)], architecture='boundary', max_len=None, error_policy='raise', build_targets=False)
    if batch.input_ids.shape[-1] > max_encoded_tokens:
        raise ValueError('Inference input exceeds the configured cap; no truncation')
    del batch
    with torch.inference_mode():
        result = model.extract_entities(text, schema['descriptions'], threshold=0., include_spans=True,
                                        include_confidence=True, max_len=None, overlap_policy='allow')
    spans = [{'label': reverse[name], 'text': s['text'], 'start': int(s['start']), 'end': int(s['end']),
              'score': float(s['confidence'])} for name, items in result['entities'].items() for s in items]
    validate_spans(text, spans, schema['names'], scored=True)
    return spans


def select(args):
    import torch
    torch.set_num_threads(args.cpu_threads)
    run = Path(args.run); meta = json.loads((run/'run.json').read_text())
    if meta['status'] != 'complete' or meta['smoke_steps']: raise ValueError('Selection requires completed full-epoch training, not a smoke run')
    schema = meta['schema']; rows = read_rows(args.dev, schema)
    if {r['id'] for r in rows}&set(meta['training_ids']): raise ValueError('Train/dev IDs overlap')
    if any(hashlib.sha256(r['text'].encode()).hexdigest() in set(meta['training_text_hashes']) for r in rows): raise ValueError('Train/dev texts overlap')
    if not any(row['spans'] for row in rows): raise ValueError('Dev must include positive gold entities')
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False); results=[]
    for name in meta['checkpoints']:
        m, recipe = load_checkpoint(run/name, args.device, args.model)
        raw = [candidates(m, r['text'], schema, recipe['max_encoded_tokens']) for r in rows]
        with (output/(name+'-candidates.jsonl')).open('x') as f:
            for row, spans in zip(rows, raw): f.write(json.dumps({'id': row['id'], 'candidates': spans}, ensure_ascii=False)+'\n')
        for i in range(1,20):
            threshold=i/20; scores=exact_counts(rows,[decode(c,threshold) for c in raw]); results.append({'checkpoint':name,'epoch':recipe['epoch'],'threshold':threshold,**scores})
        del m;gc.collect()
        if args.device=='cuda':__import__('torch').cuda.empty_cache()
    best=max(results,key=lambda x:(x['f1'] if x['f1'] is not None else -1,-x['epoch'],x['threshold']))
    save(output/'scores.json',results)
    import os
    save(output/'selection.json',{'checkpoint':os.path.relpath(run/best['checkpoint'],output),'threshold':best['threshold'],
         'dev_sha256':sha(args.dev),'criterion':'max exact micro-F1, then earlier epoch, then higher threshold',
         'checkpoint_recipe_sha256':sha(run/best['checkpoint']/'recipe.json'),'metrics':best})
    print(json.dumps(best))


def predict(args):
    import torch
    torch.set_num_threads(args.cpu_threads)
    if args.selection and args.threshold is not None:raise ValueError('A dev selection already fixes the threshold')
    if args.selection:
        p=Path(args.selection); selection=json.loads(p.read_text()); checkpoint=(p.parent/selection['checkpoint']).resolve()
        if sha(checkpoint/'recipe.json')!=selection['checkpoint_recipe_sha256']:raise ValueError('Selected checkpoint recipe changed')
        threshold=selection['threshold']
    else:
        if args.threshold is None:raise ValueError('Explicit --threshold is required without a dev selection')
        checkpoint=Path(args.checkpoint);threshold=args.threshold
    if not math.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('Threshold outside [0,1]')
    model,recipe=load_checkpoint(checkpoint,args.device,args.model)
    text=Path(args.text_file).read_text(encoding='utf-8').rstrip('\n')
    spans=decode(candidates(model,text,recipe['schema'],recipe['max_encoded_tokens']),threshold)
    print(json.dumps({'text':text,'spans':spans,'masked':mask(text,spans),'threshold':threshold},ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--data',required=True);q.add_argument('--schema',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('train');q.add_argument('--data',required=True);q.add_argument('--schema',required=True);q.add_argument('--config',required=True);q.add_argument('--output',required=True);q.add_argument('--model');q.add_argument('--device',choices=['cpu','cuda'],default='cuda');q.add_argument('--smoke-steps',type=int,default=0);q.add_argument('--cpu-threads',type=int,default=4)
    q=sub.add_parser('select');q.add_argument('--run',required=True);q.add_argument('--dev',required=True);q.add_argument('--output',required=True);q.add_argument('--model');q.add_argument('--device',choices=['cpu','cuda'],default='cuda');q.add_argument('--cpu-threads',type=int,default=4)
    q=sub.add_parser('predict');group=q.add_mutually_exclusive_group(required=True);group.add_argument('--selection');group.add_argument('--checkpoint');q.add_argument('--threshold',type=float);q.add_argument('--model');q.add_argument('--text-file',required=True);q.add_argument('--device',choices=['cpu','cuda'],default='cpu');q.add_argument('--cpu-threads',type=int,default=4)
    args=p.parse_args()
    if hasattr(args,'cpu_threads') and args.cpu_threads<1:p.error('CPU threads must be positive')
    if args.command=='prepare':
        schema=read_schema(args.schema);rows=read_rows(args.data,schema);out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('x',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(to_example(row,schema),ensure_ascii=False)+'\n')
        print(json.dumps({'status':'validated_and_exported','documents':len(rows)}))
    elif args.command=='train':
        if args.smoke_steps<0 or args.cpu_threads<1:p.error('Invalid smoke step/thread count')
        train(args)
    elif args.command=='select':select(args)
    else:predict(args)


if __name__=='__main__':main()
