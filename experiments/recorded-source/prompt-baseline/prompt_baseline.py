"""Frozen zero-shot extraction with native causal LMs; no training or answer retries."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
LABELS=('PRIVATE_PRICE','NEGOTIATION_LIMIT','INDIVIDUAL_DISCOUNT','RETRO_BONUS','PAYMENT_TERMS','INTERNAL_MARGIN')
DATA_SHA={'train':'a8680889b69a58aee07283bfc61ef52f870f72f717d5ac1cc6fd67527c385605','dev':'5c71a99cb44b659f2fc50e959c78914e683fce204dbdfe381ebe433a5c8f231f','test':'d06f34befb2db819eef3106c1aa1e0f99f5b53c0c9690aaeabeda37026870601'}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()

def write(path,value):
 p=Path(path);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(p)

def parse_answer(text,raw):
 """Exact-only decoder. Malformed root -> empty; invalid entities are logged/dropped."""
 errors=[];pred=[]
 try:value=json.loads(raw.strip())
 except (ValueError,TypeError):return [],['invalid_json']
 if not isinstance(value,dict) or set(value)!= {'entities'} or not isinstance(value['entities'],list):return [],['invalid_schema']
 for index,item in enumerate(value['entities']):
  reason=None
  if not isinstance(item,dict) or set(item)!= {'text','label','occurrence'}:reason='entity_schema'
  elif not isinstance(item['label'],str) or item['label'] not in LABELS or not isinstance(item['text'],str) or not item['text']:reason='entity_label_or_text'
  elif type(item['occurrence']) is not int or item['occurrence']<1:reason='entity_occurrence'
  if reason:errors.append({'index':index,'reason':reason});continue
  surface=item['text'];positions=[];start=0
  while True:
   start=text.find(surface,start)
   if start<0:break
   positions.append(start);start+=1
  occurrence=item['occurrence']
  if occurrence>len(positions):errors.append({'index':index,'reason':'substring_or_occurrence_not_found'});continue
  a=positions[occurrence-1];b=a+len(surface)
  if surface[0].isspace() or surface[-1].isspace():errors.append({'index':index,'reason':'external_whitespace'});continue
  if any(a<p['end'] and p['start']<b for p in pred):errors.append({'index':index,'reason':'overlap_or_duplicate'});continue
  pred.append({'start':a,'end':b,'text':surface,'label':item['label'],'score':1.0})
 return sorted(pred,key=lambda s:(s['start'],s['end'],s['label'])),errors

def load_rows(directory,split):
 p=Path(directory)/(split+'.jsonl')
 if sha(p)!=DATA_SHA[split]:raise ValueError('Frozen data hash changed: '+split)
 rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
 if len(rows)!= {'train':5000,'dev':500,'test':1000}[split]:raise ValueError('Incomplete dataset')
 return rows

def evaluate(rows,predictions,parent_source):
 source=Path(parent_source);sys.path.insert(0,str(source))
 spec=importlib.util.spec_from_file_location('_prompt_span_evaluate',source/'evaluate.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module.evaluate(rows,predictions)

def load_model(config):
 import torch,transformers
 if not torch.cuda.is_available():raise RuntimeError('Full prompt baseline requires CUDA')
 path=Path(config['model_dir']);raw=json.loads((path/'config.json').read_text())
 family=config['family'];cfg_cls=transformers.Qwen3_5TextConfig if family=='qwen35' else transformers.Gemma4TextConfig
 cls=transformers.Qwen3_5ForCausalLM if family=='qwen35' else transformers.Gemma4ForCausalLM
 cfg=cfg_cls.from_dict(raw['text_config']);cfg.use_cache=True
 expected=json.loads(Path(config['hub_manifest']).read_text())
 if expected['key']!=config['key']:raise ValueError('Hub/model key mismatch')
 for name,info in expected['files'].items():
  artifact=path/name
  if artifact.is_file() and not name.endswith('.safetensors'):
   if artifact.stat().st_size!=info['bytes']:raise ValueError('Metadata size mismatch: '+name)
   raw_file=artifact.read_bytes();git_blob=hashlib.sha1(b'blob '+str(len(raw_file)).encode()+b'\x00'+raw_file).hexdigest()
   digest=info.get('lfs_sha256')
   if (digest and hashlib.sha256(raw_file).hexdigest()!=digest) or (not digest and git_blob!=info['blob_id']):raise ValueError('Metadata hash mismatch: '+name)
 for name,info in expected['files'].items():
  if name.endswith('.safetensors'):
   if (path/name).stat().st_size!=info['bytes'] or sha(path/name)!=info['lfs_sha256']:raise ValueError('Base weight mismatch: '+name)
 model,loading=cls.from_pretrained(str(path),config=cfg,dtype=torch.bfloat16,device_map={'':'cuda'},local_files_only=True,
   trust_remote_code=False,attn_implementation='sdpa',use_kernels=False,key_mapping={r'^model\.language_model\.':'model.'},output_loading_info=True)
 if loading['missing_keys'] or loading['mismatched_keys'] or loading['error_msgs']:raise ValueError('Incomplete text LM load: '+str(loading))
 allowed=re.compile(r'^(?:model\.)?(?:visual|audio_tower|vision_tower|embed_audio|embed_vision|mtp)(?:\.|$)|^(?:model\.)?(?:language_model\.)?layers\.(?:1[5-9]|2\d|3[0-4])\.(?:self_attn\.)?(?:k_proj|v_proj|k_norm)\.')
 unexpected=loading.get('unexpected_keys',[])
 if any(not allowed.match(k) for k in unexpected):raise ValueError('Unapproved unexpected checkpoint keys: '+str(unexpected[:12]))
 if model.lm_head.weight.data_ptr()!=model.model.embed_tokens.weight.data_ptr():raise ValueError('Expected tied pretrained LM head')
 tokenizer=transformers.AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=False,padding_side='left')
 if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
 if not tokenizer.chat_template:raise ValueError('Native chat template missing')
 generation=transformers.GenerationConfig.from_pretrained(str(path),local_files_only=True) if (path/'generation_config.json').exists() else transformers.GenerationConfig.from_model_config(cfg)
 # Qwen text_config EOS is <|endoftext|>, whereas the native chat turn ends
 # with tokenizer EOS <|im_end|>. Both are legitimate stopping tokens.
 eos=generation.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
 generation.eos_token_id=sorted({token for token in eos+[tokenizer.eos_token_id] if token is not None})
 model.generation_config=generation;model.eval()
 loading={k:sorted(v) if isinstance(v,set) else v for k,v in loading.items()}
 return model,tokenizer,{'repo_id':expected['repo_id'],'revision':expected['revision'],'loading':loading,'parameters':sum(p.numel() for p in model.parameters()),'dtype':str(next(model.parameters()).dtype),'class':cls.__name__,'lm_head_tied':True,'vision_audio_loaded':False,'chat_template_sha256':hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()}

class Generator:
 def __init__(self,model,tokenizer,prompt,config):self.model=model;self.tokenizer=tokenizer;self.prompt=prompt;self.config=config
 def __call__(self,rows,deadline):
  import torch
  from transformers import StoppingCriteria,StoppingCriteriaList
  class Deadline(StoppingCriteria):
   def __call__(self,input_ids,scores,**kwargs):return time.time()>=deadline
  torch.cuda.synchronize();started=time.perf_counter()
  texts=[self.tokenizer.apply_chat_template([{'role':'system','content':self.prompt},{'role':'user','content':r['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in rows]
  inputs=self.tokenizer(texts,return_tensors='pt',padding=True,add_special_tokens=False,truncation=False)
  lengths=inputs['attention_mask'].sum(-1).tolist()
  if max(lengths)+self.config['max_new_tokens']>self.config['max_context_tokens']:raise ValueError('No truncation: context exceeds configured limit')
  inputs={k:v.to('cuda') for k,v in inputs.items()};width=inputs['input_ids'].shape[1]
  with torch.inference_mode():outputs=self.model.generate(**inputs,max_new_tokens=self.config['max_new_tokens'],do_sample=False,use_cache=True,logits_to_keep=1,pad_token_id=self.tokenizer.pad_token_id,stopping_criteria=StoppingCriteriaList([Deadline()]))
  result=[];eos=self.model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
  for row,ids,length in zip(rows,outputs[:,width:].tolist(),lengths):
   end=next((i+1 for i,t in enumerate(ids) if t in eos),len(ids));ids=ids[:end]
   decoded=self.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
   predictions,errors=parse_answer(row['text'],decoded)
   result.append({'id':row['id'],'text':row['text'],'raw_response':decoded,'raw_token_ids':ids,'predictions':predictions,'decoder_errors':errors,'input_tokens':length,'output_tokens':len(ids),'terminated_eos':bool(ids and ids[-1] in eos),'truncated':bool(ids and ids[-1] not in eos)})
  torch.cuda.synchronize();return result,time.perf_counter()-started

def append_records(output,phase,records,start):
 directory=output/phase;directory.mkdir(exist_ok=True)
 for index,record in enumerate(records,start):
  p=directory/f'{index:04d}.json'
  if p.exists():raise FileExistsError('Record already exists; no answer retries')
  with p.open('x') as f:json.dump(record,f,ensure_ascii=False);f.write('\n');f.flush();__import__('os').fsync(f.fileno())

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--deadline-unix',type=float);args=p.parse_args()
 cfg=json.loads(Path(args.config).read_text());output=Path(args.output);output.mkdir(parents=True,exist_ok=False);write(output/'config.json',cfg)
 deadline=args.deadline_unix or time.time()+cfg.get('maximum_seconds',1200)
 dev=load_rows(cfg['data_dir'],'dev');test=load_rows(cfg['data_dir'],'test');prompt=Path(cfg['prompt']).read_text();write(output/'freeze.json',{'prompt_sha256':sha(cfg['prompt']),'config_sha256':sha(args.config),'source_sha256':sha(__file__),'data_hashes':DATA_SHA,'deadline_unix':deadline,'test_used_for_tuning':False,'decode_policy':'strict JSON root; invalid entities dropped with diagnostics; exact substring plus one-based occurrence; no retries'})
 try:
  model,tokenizer,audit=load_model(cfg);write(output/'model-audit.json',audit);generator=Generator(model,tokenizer,prompt,cfg)
  import numpy as np
  import torch
  write(output/'hardware.json',{'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'torch':torch.__version__,'transformers':__import__('transformers').__version__,'parameter_dtype':str(next(model.parameters()).dtype)})
  pilot=[];pilot_seconds=0
  for start in range(0,20,2):
   if time.time()>=deadline:raise TimeoutError('Deadline during dev pilot')
   records,seconds=generator(dev[start:start+2],deadline);pilot+=records;pilot_seconds+=seconds;append_records(output,'dev-pilot',records,start)
  write(output/'dev-pilot.json',{'documents':20,'seconds':pilot_seconds,'output_tokens':sum(r['output_tokens'] for r in pilot),'decoder_error_documents':sum(bool(r['decoder_errors']) for r in pilot),'estimated_quality_seconds':pilot_seconds/20*1000,'metrics':evaluate(dev[:20],[{'id':r['id'],'text':r['text'],'predictions':r['predictions']} for r in pilot],cfg['parent_source'])})
  estimate=pilot_seconds/20*1000*1.5+120
  write(output/'resource-decision.json',{'estimated_full_quality_plus_latency_seconds':estimate,'remaining_seconds':deadline-time.time(),'action':'run full quality and fixed200 latency if deadline permits','criterion':'Resource estimate only; prompt and generation limits remain frozen regardless of dev quality'})
  for start in range(20,40,2):
   if time.time()>=deadline:raise TimeoutError('Deadline during throughput warmup')
   warm,_=generator(dev[start:start+2],deadline);append_records(output,'throughput-warmup',warm,start)
  quality=[];processing_seconds=0;wall=time.perf_counter()
  for start in range(0,1000,2):
   if time.time()>=deadline:break
   records,seconds=generator(test[start:start+2],deadline);processing_seconds+=seconds;quality+=records;append_records(output,'test-quality',records,start)
   write(output/'progress.json',{'phase':'test-quality','documents':len(quality),'processing_seconds':processing_seconds,'time_unix':time.time()})
  write(output/'quality-pass.json',{'documents':len(quality),'processing_seconds':processing_seconds,'wall_seconds_including_durable_io':time.perf_counter()-wall,'batch_size':2})
  if len(quality)!=1000:raise TimeoutError('Partial test preserved; do not report as complete')
  predictions=[{'id':r['id'],'text':r['text'],'predictions':r['predictions']} for r in quality]
  with (output/'predictions.jsonl').open('x') as stream:
   for row in predictions:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
  metrics=evaluate(test,predictions,cfg['parent_source']);write(output/'metrics.json',metrics)
  if time.time()<deadline:
   for index,row in enumerate(dev[20:40]):
    if time.time()>=deadline:break
    warm,_=generator([row],deadline);append_records(output,'latency-warmup',warm,index)
  latency=[];sample=[test[int(i)] for i in np.linspace(0,999,200)]
  for index,row in enumerate(sample):
   if time.time()>=deadline:break
   records,seconds=generator([row],deadline);record={**records[0],'seconds':seconds};latency.append(record);append_records(output,'latency-records',[record],index)
   write(output/'progress.json',{'phase':'latency','documents':len(latency),'time_unix':time.time()})
  with (output/'latency.jsonl').open('x') as stream:
   for row in latency:stream.write(json.dumps({'id':row['id'],'seconds':row['seconds'],'output_tokens':row['output_tokens']})+'\n')
  if len(latency)!=200:raise TimeoutError('Partial latency preserved; complete speed unavailable')
  times=[r['seconds'] for r in latency];benchmark={'latency_documents':200,'latency_batch_size':1,'latency_p50_ms':float(np.percentile(times,50))*1000,'latency_p95_ms':float(np.percentile(times,95))*1000,'throughput_documents':1000,'throughput_batch_size':2,'throughput_seconds':processing_seconds,'documents_per_second':1000/processing_seconds,'scope':'chat template, tokenization, CUDA transfer, greedy generation, decode and exact-source validation; excludes model loading and durable IO','quality_and_throughput_same_pass':True,'warmup':{'batch2_calls':20,'batch1_calls':20,'source':'dev'},'gpu':torch.cuda.get_device_name(),'model_key':cfg['key'],'quality_output_tokens':sum(r['output_tokens'] for r in quality),'latency_output_tokens':sum(r['output_tokens'] for r in latency),'invalid_documents':sum(bool(r['decoder_errors']) for r in quality),'truncated_documents':sum(r['truncated'] for r in quality)}
  write(output/'benchmark.json',benchmark);write(output/'complete.json',{'status':'complete','metrics':{k:v for k,v in metrics.items() if k!='document_details'},'benchmark':benchmark,'model':audit})
 except BaseException as error:
  write(output/'failure.json',{'type':type(error).__name__,'message':str(error),'time_unix':time.time()});raise

if __name__=='__main__':main()
