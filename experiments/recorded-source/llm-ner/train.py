"""Budget-bounded LoRA+BIO training, dev selection and frozen test evaluation."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import BIO_LABELS, DataCollator, encode_row, read_rows
from evaluate import decode_probabilities, evaluate
from models import build_model, save_adapter


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2,
                     default=lambda x: x.item() if hasattr(x, 'item') else str(x), allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)
    fsync_directory(path.parent)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sync():
    torch.cuda.synchronize()


def ensure_time(cfg, reserve_seconds=15):
    if time.time()+reserve_seconds >= cfg['deadline_unix']:
        raise TimeoutError('Stage deadline reached; preserve results and do not start new work')


# Original corpus-v1 pins, copied from CRM's frozen manifest. Runtime configs
# cannot replace these with the hash of whichever files happen to be supplied.
CORPUS_COUNTS = {'train':5000, 'dev':500, 'test':1000}
CORPUS_SHA256 = {
    'train':'a8680889b69a58aee07283bfc61ef52f870f72f717d5ac1cc6fd67527c385605',
    'dev':'5c71a99cb44b659f2fc50e959c78914e683fce204dbdfe381ebe433a5c8f231f',
    'test':'d06f34befb2db819eef3106c1aa1e0f99f5b53c0c9690aaeabeda37026870601',
}
CORPUS_MANIFEST_SHA256 = '8f780ea3535d26099257556dbe201cc2ab15aa5fda8c1100aa62f0cde4b35ddf'
SCHEMA_SHA256 = 'd5e739424b0ef717e41cdae8cc44250ca5c9ac9e73a9472c731f812a879804f7'


def frozen_inputs(cfg):
    directory=Path(cfg['data_dir'])
    manifest=directory/'manifest.json'
    if sha(manifest)!=CORPUS_MANIFEST_SHA256:
        raise ValueError('Original corpus manifest changed')
    hashes={};counts={}
    for split in ('train','dev','test'):
        path=directory/f'{split}.jsonl';hashes[split]=sha(path)
        if hashes[split]!=CORPUS_SHA256[split]:
            raise ValueError(f'Original corpus SHA changed: {split}')
        with path.open() as stream: counts[split]=sum(bool(line.strip()) for line in stream)
        if counts[split]!=CORPUS_COUNTS[split]:
            raise ValueError(f'Original corpus count differs: {split}')
    schema=directory/'model-label-schema.json'
    if sha(schema)!=SCHEMA_SHA256:raise ValueError('Original schema changed')
    return {'config':cfg,'data':hashes,'document_counts':counts,
            'corpus_manifest_sha256':CORPUS_MANIFEST_SHA256,'schema_sha256':SCHEMA_SHA256,
            'source':{name:sha(Path(__file__).with_name(name)) for name in ('train.py','data.py','evaluate.py','models.py')}}


def fsync_directory(directory):
    fd=os.open(str(directory),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def atomic_numpy(path, values):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('xb') as stream:
        np.savez_compressed(stream,**values);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path);fsync_directory(path.parent)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def selection_key(result):
    return (result['selected']['micro']['f1'] or 0.,
            -result['selected']['public_all']['masked'], -result['epoch'], result['selected']['threshold'])


def validate_selection(selection_path,cfg,current_inputs):
    """Require a complete main run and its exact selected complete checkpoint."""
    selection_path=Path(selection_path).resolve()
    if selection_path.name!='complete.json':raise ValueError('Select the main run complete.json')
    run=selection_path.parent
    training_inputs=json.loads((run/'frozen-inputs.json').read_text())
    if current_inputs['data']!=training_inputs['data'] or current_inputs['source']!=training_inputs['source']:
        raise ValueError('Runtime code/data differ from the selected training run')
    for key in ('model_key','model_dir','repeat','max_length','rank','alpha','dropout','seed','eval_batch_size','thresholds'):
        if cfg[key]!=training_inputs['config'][key]:raise ValueError('Selected model/preprocessing changed: '+key)
    completed=json.loads(selection_path.read_text())
    if completed.get('status')!='complete' or completed.get('pilot') is not False:
        raise ValueError('Evaluation requires a complete non-pilot main run')
    original_cfg=training_inputs['config'];epochs=original_cfg['epochs'];count=CORPUS_COUNTS['train']
    if type(epochs) is not int or epochs<1:raise ValueError('Invalid main epoch count')
    expected_micro=sum(math.ceil(min(128,count-start)/original_cfg['micro_batch_size']) for start in range(0,count,128))
    expected_steps=math.ceil(expected_micro/original_cfg['gradient_accumulation'])
    if (completed.get('examples_seen')!=count*epochs or completed.get('optimizer_steps')!=expected_steps*epochs
            or completed.get('completed_epochs')!=epochs):
        raise ValueError('Main run did not complete every original train row/epoch')
    selections=json.loads((run/'selections.json').read_text())
    if len(selections)!=epochs or {x['epoch'] for x in selections}!=set(range(1,epochs+1)):
        raise ValueError('Incomplete or duplicate epoch dev selections')
    frozen_sha=sha(run/'frozen-inputs.json')
    for item in selections:
        epoch=item['epoch'];checkpoint=Path(item['checkpoint']).resolve()
        if checkpoint!=(run/f'epoch-{epoch}').resolve() or not checkpoint.is_relative_to(run):
            raise ValueError('Checkpoint does not belong to this main epoch')
        state_path=checkpoint/'checkpoint-state.json'
        if sha(state_path)!=item.get('checkpoint_state_sha256'):raise ValueError('Checkpoint manifest changed after dev selection')
        state=json.loads(state_path.read_text())
        if (state.get('status')!='complete' or state.get('pilot') is not False or state.get('epoch_complete') is not True
                or state.get('epoch')!=epoch or state.get('examples_seen')!=count*epoch
                or state.get('optimizer_steps')!=expected_steps*epoch
                or state.get('training_config_sha256')!=canonical_sha(original_cfg)
                or state.get('frozen_inputs_sha256')!=frozen_sha):
            raise ValueError('Checkpoint is partial, from a pilot, or from another main run')
        for relative,info in state['files'].items():
            path=(checkpoint/relative).resolve()
            if not path.is_relative_to(checkpoint) or not path.is_file() or path.stat().st_size!=info['bytes'] or sha(path)!=info['sha256']:
                raise ValueError('Checkpoint artifact changed: '+relative)
        if 'training-state.pt' not in state['files'] or 'ner-adapter.json' not in state['files']:
            raise ValueError('Checkpoint is missing adapter or optimizer/RNG provenance')
        if [x['threshold'] for x in item['threshold_grid']]!=original_cfg['thresholds']:
            raise ValueError('Incomplete threshold sweep')
        best=max(item['threshold_grid'],key=lambda x:(x['micro']['f1'] or 0.,-x['public_all']['masked'],x['threshold']))
        if item['selected']!=best:raise ValueError('Checkpoint threshold differs from the recorded dev optimum')
        dev_path=run/f'dev-epoch-{epoch}.json'
        if json.loads(dev_path.read_text())!=item:raise ValueError('Epoch dev evidence changed')
    if completed.get('selected')!=max(selections,key=selection_key):
        raise ValueError('Main selected checkpoint differs from the recorded dev optimum')
    return completed


def autocast():
    return torch.autocast('cuda', dtype=torch.bfloat16)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def hardware():
    versions = {}
    for name in ('torch', 'transformers', 'peft', 'tokenizers', 'numpy', 'flash-linear-attention', 'causal-conv1d'):
        try: versions[name] = version(name)
        except PackageNotFoundError: versions[name] = None
    return {'created_utc': datetime.now(timezone.utc).isoformat(), 'versions': versions,
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
            'capability': list(torch.cuda.get_device_capability()),
            'memory_bytes': torch.cuda.get_device_properties(0).total_memory}


def encode(rows, tokenizer, cfg):
    return [encode_row(row, tokenizer, repeat=cfg['repeat'], max_length=cfg['max_length']) for row in rows]


def batches(examples, batch_size, seed, epoch=0):
    indices = list(range(len(examples)))
    rng = random.Random(seed + epoch)
    rng.shuffle(indices)
    # Shuffle first, then bucket nearby lengths to avoid padding every short
    # document to the length of an unrelated long document.
    groups = []
    for start in range(0, len(indices), 128):
        block = sorted(indices[start:start+128], key=lambda i: len(examples[i]['input_ids']))
        groups.extend(block[i:i+batch_size] for i in range(0, len(block), batch_size))
    rng.shuffle(groups)
    return groups


def tensors(collator, encoded):
    return {key: value.to('cuda') for key, value in collator(encoded).items()
            if key in ('input_ids', 'attention_mask', 'labels')}


def predict(model, tokenizer, rows, cfg, batch_size, output=None):
    model.eval()
    collator = DataCollator(tokenizer)
    all_encoded, all_probabilities = [], []
    started = time.perf_counter()
    path=Path(output) if output else None
    index=None
    if path:
        if path.exists() or path.with_suffix('.index.json').exists() or path.with_suffix('.chunks').exists():
            raise FileExistsError('Prediction output already exists; preserve the previous attempt')
        path.parent.mkdir(parents=True,exist_ok=True)
        chunks=path.with_suffix('.chunks');chunks.mkdir()
        index={'status':'running','ids':[],'expected_ids':[r['id'] for r in rows],
               'probability_dtype':'float32','labels':list(BIO_LABELS),'repeat':cfg['repeat'],
               'preprocessing':'exact encoding saved with this run','chunks':[],
               'completed_documents':0,'expected_documents':len(rows)}
        write_json(path.with_suffix('.index.json'),index)
    try:
        with torch.inference_mode(), autocast():
            for offset in range(0,len(rows),batch_size):
                ensure_time(cfg)
                batch_rows=rows[offset:offset+batch_size]
                encoded=encode([{'id':r['id'],'text':r['text']} for r in batch_rows],tokenizer,cfg)
                inputs=tensors(collator,encoded);inputs.pop('labels',None)
                logits=model(**inputs).logits
                probabilities=logits.float().softmax(-1).cpu().numpy()
                if len(encoded)!=len(batch_rows) or len(probabilities)!=len(batch_rows):raise ValueError('Prediction batch length changed')
                batch_values=[probs[:len(item['input_ids'])].copy() for item,probs in zip(encoded,probabilities)]
                if path:
                    chunk=chunks/f'batch-{offset:06d}.npz'
                    atomic_numpy(chunk,{f'p{offset+i:04d}':value for i,value in enumerate(batch_values)})
                    # Exact source offsets are sufficient to reconstruct candidates
                    # without another neural call, including after a partial run.
                    mapping=chunks/f'batch-{offset:06d}.json'
                    write_json(mapping,{'records':[{'id':row['id'],'text':row['text'],
                        'input_ids':item['input_ids'],'attention_mask':item['attention_mask'],
                        'metadata':item.get('metadata',{'text':row['text']}),
                        'offsets':item.get('offsets'),'prediction_mask':item.get('prediction_mask'),
                        'sequence_tokens':len(item['input_ids'])} for row,item in zip(batch_rows,encoded)]})
                    index['chunks'].append({'start':offset,'count':len(batch_rows),
                        'npz':str(chunk.relative_to(path.parent)),'sha256':sha(chunk),
                        'mapping':str(mapping.relative_to(path.parent)),'mapping_sha256':sha(mapping)})
                    index['ids'].extend(r['id'] for r in batch_rows);index['completed_documents']+=len(batch_rows)
                    write_json(path.with_suffix('.index.json'),index)
                all_encoded.extend(encoded);all_probabilities.extend(batch_values)
                del logits,probabilities,inputs
        sync();seconds=time.perf_counter()-started
        if path:
            atomic_numpy(path,{f'p{i:04d}':value for i,value in enumerate(all_probabilities)})
            index.update(status='complete',neural_pass_seconds=seconds,aggregate_sha256=sha(path),
                         timing_scope='preprocessing, neural pass and per-batch durable cache; excludes aggregate NPZ')
            write_json(path.with_suffix('.index.json'),index)
        return all_encoded,all_probabilities,seconds
    except BaseException as error:
        if path:
            index.update(status='partial',error_type=type(error).__name__,
                         seconds=time.perf_counter()-started)
            write_json(path.with_suffix('.index.json'),index)
        raise


def predictions(rows, encoded, probabilities, threshold):
    return [{'id': row['id'], 'text': row['text'],
             'predictions': decode_probabilities(item, probs, threshold=threshold)}
            for row, item, probs in zip(rows, encoded, probabilities)]


def tune(rows, encoded, probabilities, grid):
    records = []
    for threshold in grid:
        metrics = evaluate(rows, predictions(rows, encoded, probabilities, threshold))
        records.append({'threshold': threshold, 'micro': metrics['micro'],
                        'public_all': metrics['public_all'], 'coverage': metrics['coverage']})
    selected = max(records, key=lambda r: (r['micro']['f1'] or 0.,
                                          -r['public_all']['masked'], r['threshold']))
    return selected, records


def save_training_checkpoint(model,tokenizer,optimizer,scheduler,path,cfg,*,epoch,step,
                             examples_seen,attempted_examples,pending_examples_discarded,
                             epoch_complete,pilot,reason,run_output):
    """A completed update is durably saved even when no compute time remains.

    Budget checks reserve time before work. This final IO is deliberately not
    skipped by the compute deadline. There is no automatic optimizer resume.
    """
    path=Path(path)
    metadata={'config':cfg,'epoch':epoch,'optimizer_steps':step,'examples_seen':examples_seen,
        'attempted_examples':attempted_examples,'pending_examples_discarded':pending_examples_discarded,
        'epoch_complete':epoch_complete,'pilot':pilot,'reason':reason}
    save_adapter(model,tokenizer,path,metadata)
    state={'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
        'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
        'epoch':epoch,'step':step,'examples_seen':examples_seen,'attempted_examples':attempted_examples,
        'pending_examples_discarded':pending_examples_discarded,'epoch_complete':epoch_complete,
        'exact_resume_implemented':False}
    temporary=path/'training-state.pt.tmp'
    with temporary.open('xb') as stream:
        torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path/'training-state.pt')
    for artifact in path.rglob('*'):
        if artifact.is_file():
            with artifact.open('rb') as stream:os.fsync(stream.fileno())
    for directory in sorted((p for p in path.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
        fsync_directory(directory)
    frozen_path=Path(run_output)/'frozen-inputs.json'
    manifest={'status':'complete','epoch':epoch,'epoch_complete':epoch_complete,'pilot':pilot,
        'optimizer_steps':step,'examples_seen':examples_seen,'attempted_examples':attempted_examples,
        'pending_examples_discarded':pending_examples_discarded,'reason':reason,
        'training_config_sha256':canonical_sha(cfg),
        'frozen_inputs_sha256':sha(frozen_path) if frozen_path.is_file() else None,
        'files':{str(p.relative_to(path)):{'bytes':p.stat().st_size,'sha256':sha(p)}
                 for p in sorted(path.rglob('*')) if p.is_file()}}
    write_json(path/'checkpoint-state.json',manifest)
    fsync_directory(path.parent)
    return manifest


def train_stage(model, tokenizer, rows, cfg, output, *, pilot):
    encoded = encode(rows, tokenizer, cfg)
    lengths = [len(x['input_ids']) for x in encoded]
    write_json(output/'lengths.json', {'documents': len(rows), 'total_tokens': sum(lengths),
        'p50': float(np.percentile(lengths, 50)), 'p95': float(np.percentile(lengths, 95)), 'max': max(lengths)})
    collator = DataCollator(tokenizer)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    if hasattr(model, 'enable_input_require_grads'): model.enable_input_require_grads()
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    backbone = [p for name, p in params if '.score.' not in name]
    head = [p for name, p in params if '.score.' in name]
    if not backbone or not head: raise ValueError('Both LoRA and new score head must be trainable')
    optimizer = torch.optim.AdamW([
        {'params': backbone, 'lr': cfg['learning_rate']},
        {'params': head, 'lr': cfg['head_learning_rate']}], weight_decay=cfg['weight_decay'])
    per_epoch_steps = math.ceil(len(batches(encoded, cfg['micro_batch_size'], cfg['seed'])) / cfg['gradient_accumulation'])
    epochs = 1 if pilot else cfg['epochs']
    total_steps = cfg['pilot_steps'] if pilot else per_epoch_steps*epochs
    warmup = max(1, math.ceil(total_steps*cfg['warmup_ratio']))
    def schedule(step):
        if step < warmup: return (step+1)/warmup
        return max(0.0, .5*(1+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    weights = torch.ones(len(BIO_LABELS), device='cuda', dtype=torch.float32)
    weights[0] = cfg['outside_class_weight']
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    step, examples_seen, records, selections = 0, 0, [], []
    attempted_examples=0;pending_examples=0;discarded=0;completed_epochs=0
    epoch=0;epoch_seen=0
    dev = read_rows(Path(cfg['data_dir'])/'dev.jsonl') if not pilot else []
    log = (output/'training.jsonl').open('x')
    interrupted = False
    try:
        try:
            for epoch in range(epochs):
                model.train();epoch_seen=0
                micro=batches(encoded,cfg['micro_batch_size'],cfg['seed'],epoch)
                for start in range(0,len(micro),cfg['gradient_accumulation']):
                    ensure_time(cfg,reserve_seconds=20)
                    group=micro[start:start+cfg['gradient_accumulation']]
                    denominator=sum(sum(cfg['outside_class_weight'] if label==0 else 1.0
                        for label in encoded[i]['labels'] if label!=-100) for indices in group for i in indices)
                    if denominator<=0:raise ValueError('Empty supervised optimizer step')
                    optimizer.zero_grad(set_to_none=True)
                    sync();tick=time.perf_counter();numerator=0.0;pending_examples=0
                    for indices in group:
                        # A deadline during accumulation never starts the next
                        # microbatch; incomplete gradients are discarded below.
                        ensure_time(cfg,reserve_seconds=20)
                        inputs=tensors(collator,[encoded[i] for i in indices])
                        labels=inputs.pop('labels')
                        with autocast():logits=model(**inputs).logits
                        loss_sum=F.cross_entropy(logits.float().reshape(-1,len(BIO_LABELS)),labels.reshape(-1),
                            weight=weights,ignore_index=-100,reduction='sum')
                        if not bool(torch.isfinite(loss_sum)):raise ValueError('Nonfinite token loss')
                        (loss_sum/denominator).backward()
                        numerator+=float(loss_sum.detach())
                        attempted_examples+=len(indices);pending_examples+=len(indices)
                        del inputs,labels,logits,loss_sum
                    grad_norm=torch.nn.utils.clip_grad_norm_([p for _,p in params],cfg['max_grad_norm'])
                    if not torch.isfinite(grad_norm):raise ValueError('Nonfinite gradient norm')
                    optimizer.step();scheduler.step();sync();step+=1
                    examples_seen+=pending_examples;epoch_seen+=pending_examples;pending_examples=0
                    record={'epoch':epoch+1,'step':step,'loss':numerator/denominator,
                        'seconds':time.perf_counter()-tick,'examples_seen':examples_seen,
                        'grad_norm':float(grad_norm),'lr':scheduler.get_last_lr(),
                        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                        'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
                    log.write(json.dumps(record)+'\n');log.flush();os.fsync(log.fileno());records.append(record)
                    if step==1 or step%10==0:print(json.dumps(record),flush=True)
                    if pilot and step>=total_steps:break
                epoch_complete=epoch_seen==len(rows)
                if not pilot and not epoch_complete:raise ValueError('An epoch must present every train row exactly once')
                checkpoint=output/f'epoch-{epoch+1}'
                save_training_checkpoint(model,tokenizer,optimizer,scheduler,checkpoint,cfg,
                    epoch=epoch+1,step=step,examples_seen=examples_seen,attempted_examples=attempted_examples,
                    pending_examples_discarded=discarded,epoch_complete=epoch_complete,pilot=pilot,
                    reason='pilot_boundary' if pilot else 'complete_epoch',run_output=output)
                completed_epochs+=int(epoch_complete)
                if pilot:
                    if step!=total_steps:raise ValueError('Pilot ended before the requested optimizer steps')
                    break
                ensure_time(cfg)
                dev_encoded,dev_probs,seconds=predict(model,tokenizer,dev,cfg,
                    cfg['eval_batch_size'],output/f'dev-epoch-{epoch+1}.npz')
                chosen,grid=tune(dev,dev_encoded,dev_probs,cfg['thresholds'])
                result={'epoch':epoch+1,'checkpoint':str(checkpoint.resolve()),'selected':chosen,
                    'checkpoint_state_sha256':sha(checkpoint/'checkpoint-state.json'),
                    'neural_dev_seconds':seconds,'threshold_grid':grid}
                selections.append(result)
                write_json(output/f'dev-epoch-{epoch+1}.json',result)
                write_json(output/'selections.json',selections)
                del dev_encoded,dev_probs
        except TimeoutError as error:
            interrupted=True;discarded+=pending_examples
            optimizer.zero_grad(set_to_none=True)
            save_training_checkpoint(model,tokenizer,optimizer,scheduler,output/'interrupted',cfg,
                epoch=epoch+1,step=step,examples_seen=examples_seen,attempted_examples=attempted_examples,
                pending_examples_discarded=discarded,epoch_complete=epoch_seen==len(rows),pilot=pilot,
                reason='compute_deadline',run_output=output)
            write_json(output/'deadline.json',{'status':'deadline','optimizer_steps':step,
                'attempted_examples':attempted_examples,'examples_applied':examples_seen,
                'pending_examples_discarded':discarded,'message':str(error)})
        result={'status':'deadline' if interrupted else 'complete','pilot':pilot,
            'seconds':time.perf_counter()-started,'optimizer_steps':step,'examples_seen':examples_seen,
            'attempted_examples':attempted_examples,'pending_examples_discarded':discarded,
            'completed_epochs':completed_epochs,'per_epoch_steps':per_epoch_steps,
            'median_step_seconds':float(np.median([r['seconds'] for r in records[3:]] or [r['seconds'] for r in records])) if records else None,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        if selections:result['selected']=max(selections,key=selection_key)
        write_json(output/'complete.json',result)
        return result
    finally:
        log.close()


def benchmark(model, tokenizer, test, dev, cfg, threshold, output):
    sample = [test[int(i)] for i in np.linspace(0,len(test)-1,200)]
    # Same fixed documents for both LLMs and the GLiNER repeat benchmark.
    for row in dev[:20]:
        encoded, probs, _ = predict(model, tokenizer, [row], cfg, 1)
        predictions([row], encoded, probs, threshold)
    times = []
    with (output/'latency.jsonl').open('x') as stream:
        for row in sample:
            sync(); started = time.perf_counter()
            encoded, probs, _ = predict(model, tokenizer, [row], cfg, 1)
            result = predictions([row], encoded, probs, threshold)
            sync(); seconds = time.perf_counter()-started
            times.append(seconds)
            stream.write(json.dumps({'id':row['id'],'seconds':seconds,'spans':len(result[0]['predictions'])})+'\n')
            stream.flush()
    for start in range(0,40,2): predict(model,tokenizer,dev[start:start+2],cfg,2)
    sync(); started = time.perf_counter(); count = 0
    for start in range(0,len(test),2):
        rows = test[start:start+2]
        encoded, probs, _ = predict(model,tokenizer,rows,cfg,2)
        count += len(predictions(rows,encoded,probs,threshold))
    sync(); elapsed = time.perf_counter()-started
    return {'latency_documents':len(sample),'latency_p50_ms':float(np.percentile(times,50))*1000,
            'latency_p95_ms':float(np.percentile(times,95))*1000,'latency_batch_size':1,
            'throughput_batch_size':2,'throughput_documents':count,'throughput_seconds':elapsed,
            'documents_per_second':count/elapsed,
            'scope':'original text -> preprocessing/tokenization, CUDA transfer, model, source spans; loading and literal mask replacement excluded'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--stage',choices=('pilot','train','evaluate'),required=True)
    parser.add_argument('--selection')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    output = Path(args.output); output.mkdir(parents=True,exist_ok=False)
    write_json(output/'config.json',cfg)
    write_json(output/'invocation.json',vars(args))
    frozen = frozen_inputs(cfg)
    write_json(output/'frozen-inputs.json',frozen)
    selected=None
    if args.stage=='evaluate':
        if not args.selection:raise ValueError('Completed main selection required for evaluation')
        selected=validate_selection(args.selection,cfg,frozen)
    elif args.selection:
        raise ValueError('Pilot/main must start from pretrained base, not a selected adapter')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Supported BF16 CUDA GPU required; full CPU training forbidden')
    seed_all(cfg['seed'])
    write_json(output/'hardware.json',hardware())
    write_json(output/'source-hashes.json',{p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    try:
        ensure_time(cfg, reserve_seconds=120)
        adapter = selected['selected']['checkpoint'] if selected else None
        model, tokenizer, audit = build_model(cfg['model_key'],cfg['model_dir'], device='cuda',
            dtype='bfloat16',rank=cfg['rank'],alpha=cfg['alpha'],dropout=cfg['dropout'],adapter_dir=adapter)
        write_json(output/'model-audit.json',audit)
        if args.stage in ('pilot','train'):
            rows = read_rows(Path(cfg['data_dir'])/'train.jsonl')
            result = train_stage(model,tokenizer,rows,cfg,output,pilot=args.stage=='pilot')
        else:
            if not selected: raise ValueError('Frozen dev selection required for test')
            threshold = selected['selected']['selected']['threshold']
            write_json(output/'frozen-selection.json',selected)
            write_json(output/'selection-reference.json',{'path':str(Path(args.selection).resolve()),'sha256':sha(args.selection)})
            test = read_rows(Path(cfg['data_dir'])/'test.jsonl')
            dev = read_rows(Path(cfg['data_dir'])/'dev.jsonl')
            encoded,probs,seconds = predict(model,tokenizer,test,cfg,cfg['eval_batch_size'],output/'test-probabilities.npz')
            result_predictions = predictions(test,encoded,probs,threshold)
            with (output/'predictions.jsonl').open('x') as stream:
                for row in result_predictions: stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            metrics = evaluate(test,result_predictions)
            write_json(output/'metrics.json',metrics)
            del encoded,probs
            timings = benchmark(model,tokenizer,test,dev,cfg,threshold,output)
            write_json(output/'benchmark.json',timings)
            result = {'status':'complete','threshold':threshold,'metrics':{k:v for k,v in metrics.items() if k!='document_details'},
                      'test_neural_seconds':seconds,'benchmark':timings}
            write_json(output/'complete.json',result)
        print(json.dumps(result,ensure_ascii=False,default=str),flush=True)
    except BaseException as error:
        write_json(output/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise


if __name__=='__main__':
    main()
