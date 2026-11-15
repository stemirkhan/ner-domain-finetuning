"""One frozen model/mode, batch1 long-document inference; never rents or trains."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import gc
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

NER_KEYS=('base','full','lora','dora','rslora','qlora','distilled_full','qwen35','gemma4')
PROMPT_KEYS=('qwen35_prompt','gemma4_prompt')
LABELS=('PRIVATE_PRICE','NEGOTIATION_LIMIT','INDIVIDUAL_DISCOUNT','RETRO_BONUS','PAYMENT_TERMS','INTERNAL_MARGIN')
TIERS=(1000,2000,4000,8000)
SCOPE=('original text -> preprocessing/tokenization, CUDA, complete model inference or generation, '
       'native chunking/merge when selected, and validated original-source spans; loading and durable IO excluded')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8388608),b''):digest.update(block)
    return digest.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temp.replace(path)


def load_source(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise ImportError(path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    try:spec.loader.exec_module(module)
    except BaseException:sys.modules.pop(name,None);raise
    return module


def resolve(root,value):
    p=Path(value);return p.resolve() if p.is_absolute() else (root/p).resolve()


def configuration(path,root_override=None):
    path=Path(path).resolve();cfg=json.loads(path.read_text())
    root=Path(root_override).resolve() if root_override else resolve(path.parent.parent,cfg.get('root','.'))
    result={**cfg,'root':str(root),'config_path':str(path),'config_sha256':sha(path)}
    for key,default in [('model_paths','configs/model-paths.json'),('helper_dir','parent-src'),
                        ('corpus_manifest','data/long-context/corpus-v1/manifest.json')]:
        result[key]=str(resolve(root,cfg.get(key,default)))
    result.setdefault('max_ner_input_tokens',32768);result.setdefault('max_prompt_context_tokens',32768)
    result.setdefault('chunk_size_words',384);result.setdefault('chunk_overlap_words',64)
    for key in ('max_ner_input_tokens','max_prompt_context_tokens'):
        if type(result[key]) is not int or result[key]<1:raise ValueError('Invalid token limit: '+key)
    if not 0<=result['chunk_overlap_words']<result['chunk_size_words']:raise ValueError('Invalid chunk overlap')
    return result


def validate_spans(text,spans,*,scored=False):
    if not isinstance(spans,list):raise ValueError('Spans must be a list')
    previous=0;seen=set()
    for s in sorted(spans,key=lambda v:(v['start'],v['end'],v['label'])):
        a,b=s['start'],s['end'];key=(s['label'],a,b)
        if (type(a) is not int or type(b) is not int or not 0<=a<b<=len(text)
                or a<previous or s['label'] not in LABELS or text[a:b]!=s['text'] or key in seen):
            raise ValueError('Invalid source span')
        if scored and (type(s.get('score')) not in (int,float) or not math.isfinite(s['score']) or not 0<=s['score']<=1):
            raise ValueError('Invalid score')
        previous=b;seen.add(key)


def load_rows(path,expected=None):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if expected is not None and len(rows)!=expected:raise ValueError(f'Expected {expected} documents, found {len(rows)}')
    if len({r.get('id') for r in rows})!=len(rows):raise ValueError('Duplicate corpus ID')
    for row in rows:
        if not isinstance(row.get('id'),str) or not isinstance(row.get('text'),str):raise ValueError('Invalid id/text')
        if type(row.get('length_tier')) is not int or row['length_tier'] not in TIERS:raise ValueError('Unknown integer length_tier')
        validate_spans(row['text'],row.get('spans',[]))
    return rows


def warmup_rows(rows):
    groups=defaultdict(list)
    for index,row in enumerate(rows):groups[row['length_tier']].append((index,row))
    if set(groups)!=set(TIERS) or any(len(groups[t])<2 for t in TIERS):
        raise ValueError('Separate pilot requires at least two texts per tier')
    return [item for tier in TIERS for item in groups[tier][:2]]


def memory_snapshot(torch):
    free,total=torch.cuda.mem_get_info()
    return {'peak_allocated_bytes':int(torch.cuda.max_memory_allocated()),
            'peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),
            'allocated_bytes':int(torch.cuda.memory_allocated()),'reserved_bytes':int(torch.cuda.memory_reserved()),
            'device_free_bytes':int(free),'device_total_bytes':int(total),'device_used_bytes':int(total-free)}


class Engine:
    """Original algorithms plus instrumentation; only bare text enters predict."""
    def __init__(self,cfg,key,mode):
        import torch
        from importlib.metadata import version
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 CUDA required; no full CPU inference')
        if key not in NER_KEYS+PROMPT_KEYS:raise ValueError('Unknown model')
        if mode not in ('full_document','word_chunks') or mode=='word_chunks' and key not in NER_KEYS[:7]:
            raise ValueError('Word chunks are supported only by GLiNER variants')
        self.torch=torch;self.cfg=cfg;self.key=key;self.mode=mode;self.runtime=None;self.predictor=None;self.hook=None
        self.model=None;self.tokenizer=None;self.encoder_lengths=[];self.last_input_info={};self.audit={}
        dependencies={name:version(name) for name in ('torch','transformers','peft','gliner2','numpy')}
        for name,expected in [('transformers','5.22.1'),('peft','0.21.2'),('gliner2','2.1.0')]:
            if dependencies[name]!=expected:raise ValueError('Frozen dependency differs: '+name)
        if key=='qlora':
            dependencies['bitsandbytes']=version('bitsandbytes')
            if dependencies['bitsandbytes']!='0.50.2':raise ValueError('Frozen bitsandbytes differs')
        helpers=Path(cfg['helper_dir']);root=Path(cfg['root'])
        for filename,field in [('replay_models.py','replay_helper_sha256'),('prompt_baseline.py','prompt_helper_sha256')]:
            if cfg.get(field) and sha(helpers/filename)!=cfg[field]:raise ValueError('Frozen helper SHA differs: '+filename)
        if cfg.get('model_paths_sha256') and sha(cfg['model_paths'])!=cfg['model_paths_sha256']:raise ValueError('Model paths SHA differs')
        self.replay=load_source('_long_replay_tools_'+sha(helpers/'replay_models.py')[:12],helpers/'replay_models.py')
        self.sources={'replay_helper':{'path':str(helpers/'replay_models.py'),'sha256':sha(helpers/'replay_models.py')}}
        started=time.perf_counter()
        if key in PROMPT_KEYS:
            source=helpers/'prompt_baseline.py'
            self.prompt_module=load_source('_long_frozen_prompt_'+sha(source)[:12],source)
            original=json.loads(resolve(root,cfg.get('prompt_configs',{}).get(key,'configs/generative-'+key.split('_')[0]+'.json')).read_text())
            self.prompt_cfg={**original,'max_context_tokens':cfg['max_prompt_context_tokens']}
            for name in ('model_dir','hub_manifest','data_dir','parent_source','prompt'):
                self.prompt_cfg[name]=str(resolve(root,self.prompt_cfg[name]))
            if self.prompt_cfg.get('max_new_tokens')!=384 or self.prompt_cfg.get('training') is not False:
                raise ValueError('Frozen generation cap or no-training policy changed')
            self.model,self.tokenizer,self.audit=self.prompt_module.load_model(self.prompt_cfg)
            self.hook=self.model.register_forward_pre_hook(self._encoder_hook,with_kwargs=True)
            prompt=Path(self.prompt_cfg['prompt']).read_text()
            self.generator=self.prompt_module.Generator(self.model,self.tokenizer,prompt,self.prompt_cfg)
            self.sources.update(prompt_helper={'path':str(source),'sha256':sha(source)},
                                prompt={'path':self.prompt_cfg['prompt'],'sha256':sha(self.prompt_cfg['prompt'])})
            self.selected={'kind':'prompt','config':self.prompt_cfg,'adapter_merged':False}
        else:
            _,legacy_data,_,records,runtime_paths,checked=self.replay.prepare(cfg['model_paths'],str(root),[key])
            self.selected=records[key];self.audit['frozen_artifacts']=checked
            self.runtime=self.replay.load_runtime(runtime_paths[self.selected['runtime']],self.selected['runtime'])
            if self.selected['kind']=='gliner':
                kwargs={'adapter_path':self.selected.get('adapter')}
                if self.selected['runtime']=='variants':kwargs['variant']=key
                with self.replay.aliases(self.runtime):
                    self.predictor=self.runtime['evaluate'].Predictor(legacy_data.parent.parent,self.selected['model'],'cuda','bf16',**kwargs)
                self.model=self.predictor.model
                actual=getattr(self.predictor,'inference_policy',None)
                if actual:
                    for name in ('adapter_merged','compute_precision','quantized_linear_modules','remaining_adapter_modules'):
                        if name in self.selected['inference_policy'] and actual.get(name)!=self.selected['inference_policy'][name]:raise ValueError('Frozen inference policy changed: '+name)
                self.hook=self.model.encoder.register_forward_pre_hook(self._encoder_hook,with_kwargs=True)
                self.audit.update(boundary_settings=self.predictor.settings,inference_policy=self.selected['inference_policy'],
                                  candidate_budget_scope='per word chunk' if mode=='word_chunks' else 'per full document',
                                  chunking='native extract_entities_long; threshold0/overlapallow then original globaldecoder' if mode=='word_chunks' else None)
            else:
                self.ner_cfg={**json.loads(Path(self.selected['config']).read_text()),'max_length':cfg['max_ner_input_tokens']}
                self.model,self.tokenizer,self.audit=self.runtime['models'].build_model(key,self.selected['model'],device='cuda',dtype='bfloat16',
                    rank=self.ner_cfg['rank'],alpha=self.ner_cfg['alpha'],dropout=self.ner_cfg['dropout'],adapter_dir=self.selected['adapter'])
                self.model.eval()
        self.torch.cuda.synchronize();self.load_seconds=time.perf_counter()-started
        self.audit.update(load_seconds=self.load_seconds,model_key=key,mode=mode,
            load_seconds_scope='frozen-input checksum validation plus model initialization/loading; not isolated weight loading',
            gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,
            native_attention_config_changed=False,training=False,dependencies=dependencies,
            model_parameter_dtypes=sorted({str(p.dtype) for p in self.model.parameters()}))

    def _encoder_hook(self,module,args,kwargs):
        ids=kwargs.get('input_ids')
        if ids is None and args:ids=args[0]
        if ids is None or len(ids.shape)!=2 or ids.shape[0]!=1:raise ValueError('Expected batch1 encoder input')
        self.encoder_lengths.append(int(ids.shape[1]))

    def predict(self,text,deadline):
        """Gold and corpus metadata cannot enter this string-only interface."""
        if not isinstance(text,str):raise TypeError('Inference receives only raw text')
        self.encoder_lengths=[];self.last_input_info={'truncated':False}
        if deadline and time.time()>=deadline:raise TimeoutError('Compute deadline reached')
        if self.key in PROMPT_KEYS:
            values,_=self.generator([{'id':'inference','text':text}],deadline or time.time()+86400)
            value=values[0];self.last_input_info.update(input_tokens=int(value['input_tokens']),source_tokens=None,
                output_tokens=value['output_tokens'],input_format='native chat template + one document',max_context_tokens=self.cfg['max_prompt_context_tokens'])
            raw={k:v for k,v in value.items() if k not in ('id','text','predictions','input_tokens')}
            # Keep legacy output-truncation spelling while input_truncated remains false.
            raw['output_truncated']=raw['truncated']
            # Frozen Generator uses a stopping criterion for deadlines. A
            # non-EOS return after that deadline is conservatively incomplete,
            # even if the truncated prefix happens to parse as valid JSON.
            raw['deadline_interrupted']=bool(deadline and time.time()>=deadline and not value['terminated_eos'])
            return value['predictions'],raw
        if self.selected['kind']=='llm':
            encoded=self.runtime['data'].encode_row({'id':'inference','text':text},self.tokenizer,repeat=2,max_length=self.cfg['max_ner_input_tokens'])
            self.last_input_info.update(input_tokens=len(encoded['input_ids']),source_tokens=encoded['metadata']['source_tokens'],
                input_format='repeat2/finalcopy',repeat=2,max_input_tokens=self.cfg['max_ner_input_tokens'])
            collator=self.runtime['data'].DataCollator(self.tokenizer)
            inputs={k:v.to('cuda') for k,v in collator([encoded]).items() if k in ('input_ids','attention_mask')}
            with self.torch.inference_mode(),self.torch.autocast('cuda',dtype=self.torch.bfloat16):
                logits=self.model(**inputs).logits
            probabilities=logits[0].float().softmax(-1).cpu().numpy()
            pred=self.runtime['evaluate'].decode_probabilities(encoded,probabilities,self.selected['threshold'])
            import numpy as np
            active=np.asarray(encoded['prediction_mask'],dtype=bool);chosen=probabilities.argmax(-1)
            raw={'token_decisions':{'input_ids':np.asarray(encoded['input_ids'],dtype=np.int32)[active],
                'offsets':np.asarray([p for p,a in zip(encoded['offsets'],active) if a],dtype=np.int32),
                'bio_label_ids':chosen[active].astype(np.uint8),'selected_tag_probability':probabilities[np.arange(len(chosen)),chosen][active]},
                'label_order':list(self.runtime['data'].BIO_LABELS),'raw_token_scope':'finalcopy argmax and its actual probability; no full13-way matrix retained'}
            del inputs,logits,probabilities
            return pred,raw
        with self.torch.inference_mode(),self.torch.autocast('cuda',dtype=self.torch.bfloat16):
            if self.mode=='full_document':candidates=self.predictor.candidates([text],batch_size=1)[0]
            else:
                result=self.model.extract_entities_long(text,self.predictor.descriptions,threshold=0.0,
                    chunk_size=self.cfg['chunk_size_words'],chunk_overlap=self.cfg['chunk_overlap_words'],batch_size=1,
                    include_spans=True,include_confidence=True,overlap_policy='allow')
                candidates=[]
                for name,items in result['entities'].items():
                    if name not in self.predictor.reverse:raise ValueError('Unexpected model label')
                    for item in items:candidates.append({'label':self.predictor.reverse[name],'start':int(item['start']),
                        'end':int(item['end']),'text':item['text'],'score':float(item['confidence'])})
                self.runtime['metrics'].validate_spans(text,candidates,LABELS,scored=True)
        if not self.encoder_lengths:raise ValueError('Encoder shape instrumentation recorded no forward')
        self.last_input_info.update(input_tokens=max(self.encoder_lengths),encoder_tokens_per_forward=list(self.encoder_lengths),
            total_encoder_tokens=sum(self.encoder_lengths),encoder_forwards=len(self.encoder_lengths),
            input_format='full document + fixed schema' if self.mode=='full_document' else 'native overlapping word chunks + fixed schema',
            chunk_size_words=self.cfg['chunk_size_words'] if self.mode=='word_chunks' else None,
            chunk_overlap_words=self.cfg['chunk_overlap_words'] if self.mode=='word_chunks' else None,
            candidate_budget_scope='per chunk' if self.mode=='word_chunks' else 'per document')
        return self.runtime['metrics'].decode(candidates,self.selected['threshold']),{'candidates':candidates}

    def close(self):
        if self.hook is not None:self.hook.remove()
        if self.predictor is not None:self.predictor.close();self.predictor=None
        if hasattr(self,'generator'):self.generator=None
        self.model=None;self.tokenizer=None;gc.collect();self.torch.cuda.empty_cache()


def persist_raw(output,stem,raw):
    raw=dict(raw);directory=Path(output)/'raw';directory.mkdir(exist_ok=True)
    arrays=raw.pop('token_decisions',None)
    if arrays is not None:
        import numpy as np
        p=directory/(stem+'.tokens.npz')
        with p.open('xb') as stream:np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
        raw['raw_token_decisions']={'path':str(p.relative_to(output)),'bytes':p.stat().st_size,'sha256':sha(p),'format':'npz/no_pickle'}
    candidates=raw.pop('candidates',None)
    if candidates is not None:
        p=directory/(stem+'.candidates.json.gz')
        with p.open('xb') as stream:
            with gzip.GzipFile(fileobj=stream,mode='wb',compresslevel=3,mtime=0) as zipped:
                zipped.write(json.dumps(candidates,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode())
            stream.flush();os.fsync(stream.fileno())
        raw['raw_candidates']={'path':str(p.relative_to(output)),'bytes':p.stat().st_size,'sha256':sha(p),'count':len(candidates)}
    return raw


def actual_attempt(engine,row,index,output,phase,deadline):
    torch=engine.torch;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    record={'id':row['id'],'text':row['text'],'index':index,'phase':phase,'status':'running',
            'model_key':engine.key,'mode':engine.mode,'length_tier':row['length_tier'],'position':row.get('position'),
            'family':row.get('family'),'truncated':False,'started_unix':time.time()}
    raw={};error=None
    try:
        predictions,raw=engine.predict(row['text'],deadline);torch.cuda.synchronize()
        validate_spans(row['text'],predictions,scored=True)
        if raw.get('deadline_interrupted'):
            error={'type':'DeadlineInterrupted','message':'Non-EOS generation returned after compute deadline; raw prefix retained, not a completed answer'}
            record.update(status='error',predictions=None,error=error,raw_predictions_before_interruption=predictions)
        else:record.update(status='complete',predictions=predictions)
    except torch.cuda.OutOfMemoryError as exc:
        error={'type':type(exc).__name__,'message':str(exc)};record.update(status='oom',predictions=None,error=error)
    except TimeoutError as exc:
        error={'type':'DeadlineInterrupted','message':str(exc)};record.update(status='error',predictions=None,error=error,deadline_interrupted=True)
    except BaseException as exc:
        error={'type':type(exc).__name__,'message':str(exc)};record.update(status='error',predictions=None,error=error)
    record['seconds']=time.perf_counter()-started
    record.update(memory_snapshot(torch));record['input_info']=dict(engine.last_input_info)
    if engine.encoder_lengths:
        if engine.key in PROMPT_KEYS:
            record['input_info'].update(model_tokens_per_forward=list(engine.encoder_lengths),
                model_forwards=len(engine.encoder_lengths),input_tokens=max(engine.encoder_lengths))
        else:
            record['input_info'].update(encoder_tokens_per_forward=list(engine.encoder_lengths),
                encoder_forwards=len(engine.encoder_lengths),input_tokens=max(engine.encoder_lengths),total_encoder_tokens=sum(engine.encoder_lengths))
    record['input_tokens']=record['input_info'].get('input_tokens')
    record['input_truncated']=False
    if raw:record.update(persist_raw(output,f'{phase}-{index:04d}',raw))
    directory=Path(output)/('records' if phase=='main' else 'warmup');directory.mkdir(exist_ok=True)
    path=directory/f'{index:04d}.json'
    if path.exists():raise FileExistsError('Refusing to replace a prior answer')
    write(path,record)
    if error:gc.collect();torch.cuda.empty_cache()
    return record


def prior_unsupported(path,cfg,key,mode):
    if path is None:return {}
    d=json.loads(Path(path).read_text())
    if d.get('status') not in ('complete','complete_with_unsupported') or d.get('pilot_only') is not True:raise ValueError('Incomplete resource pilot')
    if d['model_key']!=key or d['mode']!=mode:raise ValueError('Pilot model/mode differs')
    if d.get('runner_config_sha256')!=cfg['config_sha256']:raise ValueError('Pilot configuration differs')
    return {int(k):v for k,v in d.get('unsupported_tiers',{}).items()}


def execute(cfg,key,mode,data,pilot_data,output,deadline=0,pilot_only=False,pilot_report=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    rows=load_rows(data,None if pilot_only else 1000);pilot=rows if pilot_only else load_rows(pilot_data)
    manifest_path=Path(cfg['corpus_manifest']);manifest=json.loads(manifest_path.read_text())
    if cfg.get('corpus_manifest_sha256') and sha(manifest_path)!=cfg['corpus_manifest_sha256']:raise ValueError('Corpus manifest SHA changed')
    if sha(data)!=manifest['pilot_sha256' if pilot_only else 'test_sha256']:raise ValueError('Frozen corpus data SHA changed')
    if sha(pilot_data or data)!=manifest['pilot_sha256']:raise ValueError('Frozen pilot data SHA changed')
    if not pilot_only and ({r['id'] for r in rows}&{r['id'] for r in pilot}):raise ValueError('Pilot/test IDs overlap')
    if not pilot_only and ({r['text'] for r in rows}&{r['text'] for r in pilot}):raise ValueError('Pilot/test text overlaps')
    warm=warmup_rows(pilot);write(output/'config.json',cfg)
    freeze={'data_path':str(Path(data).resolve()),'data_sha256':sha(data),'documents':len(rows),
        'pilot_data_path':str(Path(pilot_data or data).resolve()),'pilot_data_sha256':sha(pilot_data or data),
        'corpus_manifest_path':str(manifest_path),'corpus_manifest_sha256':sha(manifest_path),
        'model_key':key,'mode':mode,'pilot_only':pilot_only,'runner_config_sha256':cfg['config_sha256'],
        'runner_sha256':sha(__file__),'batch_size':1,'scope':SCOPE,'input_truncation':False,'training':False,
        'model_paths_sha256':sha(cfg['model_paths']),
        'warmup_calls_per_supported_tier':2,'deadline_unix':deadline}
    write(output/'freeze.json',freeze)
    unsupported=prior_unsupported(pilot_report,cfg,key,mode);attempts=[];warm_attempts=[];skipped=[];engine=None
    try:
        if deadline and time.time()>=deadline:raise TimeoutError('Deadline before model loading')
        engine=Engine(cfg,key,mode);write(output/'model-audit.json',engine.audit);write(output/'selected-reference.json',engine.selected)
        freeze.update(model_audit_sha256=sha(output/'model-audit.json'),
                      selected_reference_sha256=sha(output/'selected-reference.json'),helper_sources=engine.sources)
        write(output/'freeze.json',freeze)
        write(output/'hardware.json',{'gpu':engine.torch.cuda.get_device_name(),'cuda':engine.torch.version.cuda,
            'capability':list(engine.torch.cuda.get_device_capability()),'total_bytes':engine.torch.cuda.get_device_properties(0).total_memory})
        for index,row in warm:
            tier=row['length_tier']
            if tier in unsupported:continue
            result=actual_attempt(engine,row,index,output,'pilot' if pilot_only else 'warmup',deadline)
            warm_attempts.append(result)
            if result['status']=='oom':unsupported[tier]={'reason':'pilot_oom' if pilot_only else 'warmup_oom','source_id':row['id'],'record':str(('warmup/'+f'{index:04d}.json'))}
            elif result['status']!='complete':raise RuntimeError('Technical warmup failure: '+str(result.get('error')))
        write(output/'support.json',{'model_key':key,'mode':mode,'unsupported_tiers':unsupported,
            'supported_tiers':[t for t in TIERS if t not in unsupported],'runner_config_sha256':cfg['config_sha256']})
        if pilot_only:
            marker={'status':'complete' if not unsupported else 'complete_with_unsupported','model_key':key,'mode':mode,
                'pilot_only':True,'runner_config_sha256':cfg['config_sha256'],'unsupported_tiers':unsupported,
                'supported_tiers':[t for t in TIERS if t not in unsupported],'expected_documents':len(rows),
                'attempted_documents':len(warm_attempts),'completed_documents':sum(v['status']=='complete' for v in warm_attempts),
                'skipped_documents':len(rows)-len(warm_attempts),'oom_documents':sum(v['status']=='oom' for v in warm_attempts)}
            write(output/'complete.json',marker);return marker
        for index,row in enumerate(rows):
            tier=row['length_tier']
            if tier in unsupported:
                skipped.append({'id':row['id'],'index':index,'length_tier':tier,'status':'unsupported_length','reason':unsupported[tier]});write(output/'skipped.json',skipped);continue
            if deadline and time.time()>=deadline:
                skipped.extend({'id':r['id'],'index':j,'length_tier':r['length_tier'],'status':'not_started_deadline'} for j,r in enumerate(rows[index:],index));write(output/'skipped.json',skipped);break
            result=actual_attempt(engine,row,index,output,'main',deadline);attempts.append(result)
            if result['status']=='oom':unsupported[tier]={'reason':'main_oom','source_id':row['id'],'record':'records/'+f'{index:04d}.json'}
            write(output/'progress.json',{'model_key':key,'mode':mode,'attempted_documents':len(attempts),
                'completed_documents':sum(r['status']=='complete' for r in attempts),'skipped_documents':len(skipped),
                'last_index':index,'last_id':row['id'],'unsupported_tiers':unsupported,'time_unix':time.time()})
            if result.get('error',{}).get('type')=='DeadlineInterrupted':
                skipped.extend({'id':r['id'],'index':j,'length_tier':r['length_tier'],'status':'not_started_deadline'} for j,r in enumerate(rows[index+1:],index+1))
                write(output/'skipped.json',skipped);break
            if result['status']=='error':raise RuntimeError('Technical main failure: '+str(result.get('error')))
        success=[r for r in attempts if r['status']=='complete'];all_accounted=len(attempts)+len(skipped)==len(rows)
        status='complete' if len(success)==len(rows) else 'complete_with_unsupported' if all_accounted and unsupported and all(s['status']=='unsupported_length' for s in skipped) else 'partial'
        marker={'status':status,'quality_complete':len(success)==len(rows),'model_key':key,'mode':mode,
            'expected_documents':len(rows),'attempted_documents':len(attempts),'completed_documents':len(success),
            'skipped_documents':len(skipped),'oom_documents':sum(r['status']=='oom' for r in attempts),
            'unsupported_tiers':unsupported,'warmup_attempted_documents':len(warm_attempts),
            'data_sha256':freeze['data_sha256'],'batch_size':1,'finished_unix':time.time()}
        write(output/'complete.json',marker);return marker
    except BaseException as error:
        write(output/'failure.json',{'type':type(error).__name__,'message':str(error),'attempted_documents':len(attempts),
            'completed_documents':sum(r['status']=='complete' for r in attempts),'unsupported_tiers':unsupported,'time_unix':time.time()})
        raise
    finally:
        if engine is not None:engine.close()


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--root')
    p.add_argument('--model',required=True,choices=NER_KEYS+PROMPT_KEYS);p.add_argument('--mode',choices=('full_document','word_chunks'),default='full_document')
    p.add_argument('--data',required=True);p.add_argument('--pilot-data');p.add_argument('--pilot-report');p.add_argument('--output',required=True)
    p.add_argument('--deadline-unix',type=float,default=0)
    return p


def main():
    a=parser().parse_args();cfg=configuration(a.config,a.root)
    pilot=a.pilot_data or cfg.get('pilot_data')
    if not pilot:raise ValueError('Separate pilot_data required')
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    result=execute(cfg,a.model,a.mode,resolve(Path(cfg['root']),a.data),resolve(Path(cfg['root']),pilot),a.output,a.deadline_unix,pilot_report=a.pilot_report)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
