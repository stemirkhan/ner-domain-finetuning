"""Strict text-only Qwen3.5/Gemma4 token classifiers with saved LoRA+BIO head.

Public runtime API: build_model(model_key, local_snapshot, device='cuda', ...)
returns (PEFT model, tokenizer, audit). It never downloads files. Full checkpoint
CPU loading is forbidden. meta_shape_audit() reads config/header metadata only.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import struct
from typing import Any

os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
# Keep checkpoint loading sequential to avoid concurrent whole-shard CPU/GPU peaks.
os.environ.setdefault('HF_DEACTIVATE_ASYNC_LOAD','1')

import torch
from torch import nn
from transformers import (AutoTokenizer, Qwen3_5TextConfig,
    Qwen3_5TextModel, Gemma4TextConfig, Gemma4TextModel)
from transformers.modeling_outputs import TokenClassifierOutput
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict

TYPES=('PRIVATE_PRICE','NEGOTIATION_LIMIT','INDIVIDUAL_DISCOUNT','RETRO_BONUS','PAYMENT_TERMS','INTERNAL_MARGIN')
BIO_LABELS=('O',)+tuple(p+'-'+t for t in TYPES for p in ('B','I'))
ID2LABEL=dict(enumerate(BIO_LABELS));LABEL2ID={v:k for k,v in ID2LABEL.items()}
MODEL_SPECS={
 'qwen35':{'repo_id':'Qwen/Qwen3.5-4B-Base','revision':'1001bb4d826a52d1f399e183466143f4da7b741b',
   'model_type':'qwen3_5_text','source_model_type':'qwen3_5',
   'config_sha256':'ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670',
   'tensor_fingerprint':'72ef349f34eaaa19be5c98fdd81f4ddd47f7b659ccae7aba1e7b094fcac7f361',
   'targets':('q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj'),
   'ignored_prefixes':('model.visual.','visual.','mtp.','lm_head.')},
 'gemma4':{'repo_id':'google/gemma-4-E2B','revision':'433813a669639fd7eaa83155828e9000ed357d53',
   'model_type':'gemma4_text','source_model_type':'gemma4',
   'config_sha256':'e5faef0dd1a8f2437f6010721146b85433eaa90e679ef011e803c7ffefae73b8',
   'tensor_fingerprint':'d447f858b3774c954206e6a0a1327fcf85e2ba6f157875853b30b10346064196',
   'targets':('q_proj','k_proj','v_proj','o_proj'),
   'ignored_prefixes':('model.vision_tower.','model.audio_tower.','model.embed_audio.','model.embed_vision.',
      'model.audio_projector.','model.multi_modal_projector.','lm_head.')},
}
TINY_PARAMETER_LIMIT=2_000_000
TEXT_PREFIX='model.language_model.'


def require(condition,message):
 if not condition:raise ValueError(message)


def plain(value):
 if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
 if isinstance(value,(tuple,list,set)):return [plain(v) for v in value]
 if isinstance(value,Path):return str(value)
 if isinstance(value,torch.dtype):return str(value)
 return value


def write_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temporary=path.with_name(path.name+'.tmp')
 temporary.write_text(json.dumps(plain(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 temporary.replace(path)


def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()


def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def dtype_value(name):
 if isinstance(name,torch.dtype):return name
 require(name in ('bfloat16','float32','bf16','fp32'),'Only BF16/FP32 are supported')
 return torch.bfloat16 if name in ('bfloat16','bf16') else torch.float32


def family_key(value):
 aliases={'qwen':'qwen35','qwen3.5':'qwen35','gemma':'gemma4','gemma4-e2b':'gemma4'}
 value=aliases.get(value,value);require(value in MODEL_SPECS,f'Unknown model family: {value}');return value


def text_config(model_key,raw):
 family=family_key(model_key);spec=MODEL_SPECS[family]
 require(raw.get('model_type') in (spec['source_model_type'],spec['model_type']), 'Wrong checkpoint model type')
 content=copy.deepcopy(raw.get('text_config',raw))
 cls=Qwen3_5TextConfig if family=='qwen35' else Gemma4TextConfig
 config=cls.from_dict(content)
 config.num_labels=len(BIO_LABELS);config.id2label=ID2LABEL.copy();config.label2id=LABEL2ID.copy()
 config.use_cache=False
 if family=='gemma4':require(config.use_bidirectional_attention is None,'Unexpected bidirectional Gemma config')
 return config


def model_class(family):return Qwen3_5TextModel if family_key(family)=='qwen35' else Gemma4TextModel


def read_header(path):
 """Read a local safetensors header without materializing any weight tensor."""
 path=Path(path)
 with path.open('rb') as stream:
  first=stream.read(8);require(len(first)==8,'Truncated safetensors header')
  size=struct.unpack('<Q',first)[0];require(1<=size<=16*1024*1024,'Invalid safetensors header size')
  raw=stream.read(size);require(len(raw)==size,'Truncated safetensors JSON')
 value=json.loads(raw)
 return {k:v for k,v in value.items() if k!='__metadata__'}


def snapshot_headers(source):
 """Accept full local weights or header-only preflight files, never download."""
 source=Path(source);require(source.is_dir(),'Provide a prepared local snapshot directory; no implicit Hub downloads')
 index_path=source/'model.safetensors.index.json';index=json.loads(index_path.read_text()) if index_path.exists() else None
 if index:shards=sorted(set(index['weight_map'].values()))
 else:shards=['model.safetensors']
 all_headers={};owners={};header_sources={}
 for filename in shards:
  require(Path(filename).name==filename,'Unsafe shard path')
  local=source/filename;header_file=source/(filename+'.header.json')
  if local.is_file():header=read_header(local);header_sources[filename]={'kind':'local_weight_header','path':str(local)}
  elif header_file.is_file():
   header={k:v for k,v in json.loads(header_file.read_text()).items() if k!='__metadata__'}
   header_sources[filename]={'kind':'header_only','path':str(header_file),'sha256':sha256(header_file)}
  else:raise FileNotFoundError(f'No weight/header available for {filename}')
  for key,value in header.items():
   require(key not in all_headers,f'Duplicate source tensor: {key}')
   require(isinstance(value.get('shape'),list) and all(type(x) is int and x>=0 for x in value['shape']),'Invalid tensor shape')
   all_headers[key]=value;owners[key]=filename
 if index:require(index['weight_map']==owners,'Index and actual shard headers differ')
 return all_headers,owners,header_sources


def _classify_source_key(key,family,expected):
 if key.startswith(TEXT_PREFIX):return 'text',key[len(TEXT_PREFIX):]
 # Tiny fixtures and a deliberately exported text-only snapshot may omit the VLM prefix.
 if key in expected:return 'text',key
 if key.startswith('model.') and key[6:] in expected:return 'text',key[6:]
 if any(key.startswith(p) for p in MODEL_SPECS[family]['ignored_prefixes']):return 'excluded',None
 return 'unexpected',None


def meta_shape_audit(model_key,source,*,allow_tiny=False):
 """Match every text backbone key and shape against a meta-only architecture."""
 family=family_key(model_key);source=Path(source).resolve();raw=json.loads((source/'config.json').read_text())
 config=text_config(family,raw)
 with torch.device('meta'):blueprint=model_class(family)(config)
 expected={k:list(v.shape) for k,v in blueprint.state_dict().items()}
 parameter_shapes={k:list(v.shape) for k,v in blueprint.named_parameters()}
 redundant_shared_kv=set()
 if family=='gemma4':
  for i,layer in enumerate(blueprint.layers):
   if layer.self_attn.is_kv_shared_layer:
    redundant_shared_kv.update(f'layers.{i}.self_attn.{leaf}.weight' for leaf in ('k_proj','v_proj','k_norm','v_norm'))
 header,owners,header_sources=snapshot_headers(source)
 loaded={};excluded={};unexpected=[];mismatched=[]
 for source_key,info in header.items():
  kind,destination=_classify_source_key(source_key,family,expected)
  if kind=='excluded':excluded[source_key]={'shape':info['shape'],'dtype':info['dtype'],'shard':owners[source_key]};continue
  if kind=='unexpected':unexpected.append(source_key);continue
  if destination not in expected:
   if destination in redundant_shared_kv:
    excluded[source_key]={'shape':info['shape'],'dtype':info['dtype'],'shard':owners[source_key],
       'reason':'unused k/v tensor in a configured KV-sharing layer; official Gemma4TextModel does not instantiate it'}
    continue
   unexpected.append(source_key);continue
  require(destination not in loaded,f'Duplicate source mapping for {destination}')
  if info['shape']!=expected[destination]:mismatched.append({'source':source_key,'target':destination,'expected':expected[destination],'found':info['shape']})
  loaded[destination]={'source':source_key,'shape':info['shape'],'dtype':info['dtype'],'shard':owners[source_key]}
 missing=sorted(set(expected)-set(loaded))
 count=sum(math.prod(shape) for shape in parameter_shapes.values())
 require(count<=TINY_PARAMETER_LIMIT if allow_tiny else count>TINY_PARAMETER_LIMIT,'Tiny/full architecture scope mismatch')
 report={'status':'pass' if not (missing or unexpected or mismatched) else 'failed','family':family,
  'repo_id':MODEL_SPECS[family]['repo_id'],'revision':MODEL_SPECS[family]['revision'],
  'source_config_sha256':sha256(source/'config.json'),'text_config_sha256':canonical_sha(config.to_dict()),
  'text_parameters':count,'text_parameter_tensors':len(parameter_shapes),'text_state_tensors':len(expected),
  'matched_text_tensors':len(loaded),'excluded_tensors':len(excluded),'redundant_shared_kv_tensors':sum(x.get('reason','').startswith('unused k/v') for x in excluded.values()),
  'text_checkpoint_numel':sum(math.prod(v['shape']) for k,v in header.items() if k.startswith(TEXT_PREFIX)),
  'missing_keys':missing,
  'unexpected_keys':unexpected,'mismatched_shapes':mismatched,'key_mapping':loaded,'excluded':excluded,
  'header_sources':header_sources,'model_weights_loaded':False,'model_forward_calls':0,
  'tiny_random_fixture':allow_tiny,'source_tensor_fingerprint':canonical_sha({k:header[k] for k in sorted(header)})}
 del blueprint
 return report


def verify_snapshot_provenance(source,family,shards):
 """Verify pinned download provenance and stream-hash actual bytes before load."""
 source=Path(source).resolve();spec=MODEL_SPECS[family_key(family)]
 path=source/'snapshot-provenance.json'
 require(path.is_file(),'Missing verified snapshot-provenance.json; finish weight download verification first')
 provenance=json.loads(path.read_text())
 require(provenance.get('verified') is True and provenance.get('repo_id')==spec['repo_id']
     and provenance.get('revision')==spec['revision'],'Unverified or wrong repository/revision')
 weights=provenance.get('weights',{})
 require(set(weights)==set(shards),'Snapshot provenance shard list differs from checkpoint')
 for filename,info in weights.items():
  require(Path(filename).name==filename,'Unsafe provenance weight filename')
  weight=source/filename
  expected=info.get('expected_sha256')
  require(info.get('verified') is True and isinstance(expected,str) and re.fullmatch('[0-9a-f]{64}',expected)
      and info.get('sha256')==expected,'Unverified Hub LFS content hash')
  require(weight.is_file() and weight.stat().st_size==info['bytes'],'Checkpoint shard missing/truncated')
  require(sha256(weight)==expected,f'Checkpoint bytes changed: {filename}')
 return {'manifest_sha256':sha256(path),'repo_id':spec['repo_id'],'revision':spec['revision'],
    'verified':True,'weight_sha256':{name:info['sha256'] for name,info in weights.items()}}


def _check_scope(device,allow_tiny_cpu):
 device=torch.device(device)
 if device.type=='cpu':require(allow_tiny_cpu,'Full LLM weight loading on CPU is prohibited; only bounded tiny fixtures allowed')
 elif device.type=='cuda':require(torch.cuda.is_available(),'CUDA unavailable; no CPU fallback')
 else:raise ValueError('Only one CUDA GPU or explicitly bounded tiny CPU fixtures are supported')
 return device


def load_backbone(model_key,source,*,device='cuda',dtype='bfloat16',allow_tiny_cpu=False):
 """Load only text tensors, fail if any text state was missing or mismatched."""
 family=family_key(model_key);device=_check_scope(device,allow_tiny_cpu);dtype=dtype_value(dtype)
 source=Path(source).resolve();audit=meta_shape_audit(family,source,allow_tiny=allow_tiny_cpu)
 require(audit['status']=='pass',f'Checkpoint structure failed: missing={audit["missing_keys"][:3]}, unexpected={audit["unexpected_keys"][:3]}, mismatched={audit["mismatched_shapes"][:3]}')
 if allow_tiny_cpu:require(audit['text_parameters']<=TINY_PARAMETER_LIMIT,'CPU parameter cap exceeded')
 if device.type=='cuda':require(dtype==torch.bfloat16 and torch.cuda.is_bf16_supported(),'Full GPU base requires BF16 support')
 for shard in audit['header_sources']:require((source/shard).is_file(),'Header-only inspection is not a loadable checkpoint')
 if not allow_tiny_cpu:
  require(audit['source_config_sha256']==MODEL_SPECS[family]['config_sha256'], 'Config differs from the audited pinned revision')
  require(audit['source_tensor_fingerprint']==MODEL_SPECS[family]['tensor_fingerprint'], 'Weight headers differ from the audited pinned revision')
  audit['snapshot_provenance']=verify_snapshot_provenance(source,family,audit['header_sources'])
 else:
  audit['snapshot_provenance']={'verified':True,'tiny_random_fixture':True,
    'weight_sha256':{name:sha256(source/name) for name in audit['header_sources']}}

 raw=json.loads((source/'config.json').read_text());cfg=text_config(family,raw)
 # Explicit native key mapping; no vision/audio/head module is ever constructed.
 mapping={r'^model\.language_model\.':''}
 backbone,loading=model_class(family).from_pretrained(str(source),config=cfg,
  local_files_only=True,trust_remote_code=False,dtype=dtype,device_map={'':str(device)},
  key_mapping=mapping,attn_implementation='sdpa',use_kernels=False,output_loading_info=True,
  ignore_mismatched_sizes=False)
 require(not loading['missing_keys'] and not loading['mismatched_keys'] and not loading['error_msgs'],
         f'Native loader did not load all text weights: {plain(loading)}')
 unexpected=list(loading['unexpected_keys'])
 allowed=set(audit['excluded'])
 require(set(unexpected)<=allowed,f'Unapproved unexpected native load keys: {sorted(set(unexpected)-allowed)[:8]}')
 require(all(not tensor.is_meta for tensor in list(backbone.parameters())+list(backbone.buffers())),'Text parameters/buffers left on meta device')
 require(sum(p.numel() for p in backbone.parameters())==audit['text_parameters'],'Loaded parameter count differs')
 require(not any(hasattr(backbone,x) for x in ('visual','vision_tower','audio_tower','lm_head')),'Multimodal/head modules unexpectedly loaded')
 audit.update(model_weights_loaded=True,native_loading_info=plain(loading),device=str(device),dtype=str(dtype),
  loaded_parameter_count=sum(p.numel() for p in backbone.parameters()),
  loaded_parameter_dtypes=dict(__import__('collections').Counter(str(p.dtype) for p in backbone.parameters())),
  excluded_weights_materialized=False)
 return backbone,audit


class TextTokenClassifier(nn.Module):
 """A shared, non-generative wrapper. Gemma PLE always receives real input_ids."""
 config_class=__import__('transformers').PreTrainedConfig
 base_model_prefix='model'
 supports_gradient_checkpointing=True
 _supports_sdpa=True
 _supports_attention_backend=True
 _no_split_modules=['Qwen3_5DecoderLayer','Gemma4TextDecoderLayer']
 def __init__(self,backbone,*,classifier_dropout=0.0):
  config=backbone.config
  config.num_labels=len(BIO_LABELS);config.id2label=ID2LABEL.copy();config.label2id=LABEL2ID.copy();config.use_cache=False
  super().__init__()
  self.config=config
  self.name_or_path=getattr(backbone,'name_or_path','')
  self.model=backbone;self.num_labels=len(BIO_LABELS)
  self.dropout=nn.Dropout(classifier_dropout)
  self.score=nn.Linear(config.hidden_size,self.num_labels,bias=True)
  # New task head only; never call post_init() over already loaded backbone.
  nn.init.normal_(self.score.weight,mean=0.0,std=config.initializer_range);nn.init.zeros_(self.score.bias)
  device=next(backbone.parameters()).device
  self.score.to(device=device,dtype=torch.float32)
 @property
 def is_gradient_checkpointing(self):return self.model.is_gradient_checkpointing
 def get_input_embeddings(self):return self.model.get_input_embeddings()
 def enable_input_require_grads(self):return self.model.enable_input_require_grads()
 def disable_input_require_grads(self):return self.model.disable_input_require_grads()
 def set_input_embeddings(self,value):return self.model.set_input_embeddings(value)
 def gradient_checkpointing_enable(self,gradient_checkpointing_kwargs=None):
  kwargs=gradient_checkpointing_kwargs or {'use_reentrant':False}
  self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
 def gradient_checkpointing_disable(self):self.model.gradient_checkpointing_disable()
 def forward(self,input_ids=None,attention_mask=None,labels=None,position_ids=None,
             use_cache=False,inputs_embeds=None,return_dict=None,output_hidden_states=None,
             output_attentions=None,**kwargs):
  if inputs_embeds is not None:raise ValueError('Use input_ids: inputs_embeds risks expensive Gemma PLE token reconstruction')
  require(input_ids is not None,'input_ids required')
  require(not use_cache,'NER requires cache disabled')
  # PEFT TOKEN_CLS forwards token_type_ids even though these backbones do not use it.
  token_types=kwargs.pop('token_type_ids',None)
  require(token_types is None,'token_type_ids are not supported')
  kwargs.pop('num_items_in_batch',None)
  require(not kwargs,f'Unsupported forward arguments: {sorted(kwargs)}')
  output=self.model(input_ids=input_ids,attention_mask=attention_mask,position_ids=position_ids,
       use_cache=False,output_hidden_states=output_hidden_states,output_attentions=output_attentions)
  hidden=output.last_hidden_state
  head_dtype=next(self.score.parameters()).dtype
  logits=self.score(self.dropout(hidden.to(head_dtype)))
  loss=None
  if labels is not None:
   require(tuple(labels.shape)==tuple(input_ids.shape),'BIO labels must match input shape')
   valid=labels!=-100;require(bool(valid.any()),'No supervised BIO tokens in batch')
   require(bool(((labels[valid]>=0)&(labels[valid]<self.num_labels)).all()),'BIO label outside class range')
   loss=torch.nn.functional.cross_entropy(logits.float().reshape(-1,self.num_labels),labels.reshape(-1),ignore_index=-100)
  return TokenClassifierOutput(loss=loss,logits=logits,
        hidden_states=getattr(output,'hidden_states',None),attentions=getattr(output,'attentions',None))


def lora_targets(model,family):
 """Audit actual Linear modules; KV-sharing layers legitimately lack k/v."""
 family=family_key(family);allowed=MODEL_SPECS[family]['targets'];targets=[];by_kind={}
 for name,module in model.named_modules():
  leaf=name.rsplit('.',1)[-1]
  if isinstance(module,nn.Linear) and leaf in allowed and ('.self_attn.' in name or '.linear_attn.' in name):
   targets.append(name);by_kind.setdefault(leaf,[]).append(name)
 require(targets,'No supported LoRA target modules')
 for leaf in ('q_proj','o_proj'):require(leaf in by_kind,f'Missing attention target {leaf}')
 if family=='qwen35':
  for leaf in ('in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj'):
   require(leaf in by_kind,f'Missing DeltaNet target {leaf}')
 require(all(not name.startswith('score') for name in targets),'Head must be fully trained, not LoRA adapted')
 return sorted(targets),{'target_modules':sorted(targets),'by_projection':by_kind,
   'kv_sharing_layers':[name for name,module in model.named_modules() if getattr(module,'is_kv_shared_layer',False)]}


def attach_lora(classifier,model_key,*,rank=16,alpha=32,dropout=.05):
 require(rank>0 and alpha>0 and 0<=dropout<1,'Invalid LoRA configuration')
 targets,audit=lora_targets(classifier,model_key)
 model=get_peft_model(classifier,LoraConfig(task_type=TaskType.TOKEN_CLS,
   r=rank,lora_alpha=alpha,lora_dropout=dropout,target_modules=targets,
   modules_to_save=['score'],bias='none'))
 trainable={n:p.numel() for n,p in model.named_parameters() if p.requires_grad}
 require(trainable and all('lora_' in name or '.score.modules_to_save.' in name for name in trainable),'Unexpected trainable base tensors')
 require(sum('.score.modules_to_save.' in n for n in trainable)==2,'Trainable score.weight/bias missing')
 audit.update(rank=rank,alpha=alpha,dropout=dropout,modules_to_save=['score'],trainable=trainable,
  trainable_parameters=sum(trainable.values()),total_parameters=sum(p.numel() for p in model.parameters()))
 return model,audit


def _tokenizer(source):
 tokenizer=AutoTokenizer.from_pretrained(str(source),use_fast=True,local_files_only=True,trust_remote_code=False)
 require(tokenizer.is_fast,'Fast tokenizer required for exact character spans')
 if tokenizer.pad_token_id is None:
  require(tokenizer.eos_token_id is not None,'Tokenizer has neither pad nor eos token')
  tokenizer.pad_token=tokenizer.eos_token
 tokenizer.padding_side='right'
 return tokenizer


def build_model(model_key,source_dir_or_repo,*,device='cuda',dtype='bfloat16',rank=16,alpha=32,dropout=.05,adapter_dir=None):
 """Build a fresh or reloaded PEFT text classifier; source must be local."""
 family=family_key(model_key);source=Path(source_dir_or_repo).resolve()
 require(source.is_dir(),'Only a prepared local model snapshot is accepted; no automatic weight downloads')
 _check_scope(device,False)
 backbone,base_audit=load_backbone(family,source,device=device,dtype=dtype)
 classifier=TextTokenClassifier(backbone)
 tokenizer=_tokenizer(source)
 audit={'base':base_audit,'family':family,'label_order':list(BIO_LABELS),
        'dependencies':{n:importlib.metadata.version(n) for n in ('torch','transformers','peft','safetensors')},
        'base_snapshot':str(source),'repo_id':MODEL_SPECS[family]['repo_id'],'revision':MODEL_SPECS[family]['revision'],
        'model_forward_calls':0,'full_cpu_load':False,
        'inference_policy':{'adapter_merged':False,'base_dtype':str(dtype_value(dtype)),
         'score_dtype':'torch.float32','autocast':'caller controlled','generation_head':False,
         'vision_audio_loaded':False},
        'tokenizer_files':{p.name:{'sha256':sha256(p),'bytes':p.stat().st_size} for p in sorted(source.iterdir())
          if p.is_file() and ('tokenizer' in p.name or p.name in ('vocab.json','merges.txt','special_tokens_map.json','tokenizer.model','sentencepiece.bpe.model'))}}
 if adapter_dir is None:
  model,lora_audit=attach_lora(classifier,family,rank=rank,alpha=alpha,dropout=dropout)
  audit['lora']=lora_audit
 else:
  adapter_dir=Path(adapter_dir);saved=verify_adapter(adapter_dir)
  require(saved['family']==family and saved['label_order']==list(BIO_LABELS),'Adapter architecture/BIO labels differ')
  require(saved['base_identity']==base_identity(audit),'Adapter was trained with a different base snapshot')
  model=PeftModel.from_pretrained(classifier,str(adapter_dir),is_trainable=False,local_files_only=True)
  audit['lora']=saved['lora'];audit['reloaded_adapter']=str(adapter_dir.resolve())
  if (adapter_dir/'tokenizer').exists():
   tokenizer=_tokenizer(adapter_dir/'tokenizer')
  require(set(get_peft_model_state_dict(model))==set(saved['adapter_tensors']),'Reloaded adapter tensor keys differ')
 model._crm_audit=audit
 return model,tokenizer,audit


def base_identity(audit):
 base=audit['base']
 return {**{k:base[k] for k in ('family','repo_id','revision','source_config_sha256','source_tensor_fingerprint','text_parameters')},
  'weight_sha256':base['snapshot_provenance']['weight_sha256'],'tiny_random_fixture':base['tiny_random_fixture']}


def verify_adapter(directory):
 directory=Path(directory);manifest=json.loads((directory/'ner-adapter.json').read_text())
 require(manifest.get('format_version')==1 and manifest.get('status')=='complete','Incomplete adapter export')
 require('adapter_model.safetensors' in manifest.get('files',{}) and 'adapter_config.json' in manifest['files'],'Adapter manifest omits required weights/config')
 for rel,info in manifest['files'].items():
  path=(directory/rel).resolve();require(path.is_relative_to(directory.resolve()),'Unsafe adapter artifact path')
  require(path.is_file() and path.stat().st_size==info['bytes'] and sha256(path)==info['sha256'],f'Adapter artifact changed: {rel}')
 return manifest


def save_adapter(model,tokenizer,output_dir,metadata=None):
 """Write-once PEFT adapter + full score head + exact tokenizer/config metadata."""
 output=Path(output_dir);require(not output.exists(),f'Adapter destination exists: {output}')
 audit=getattr(model,'_crm_audit',None);require(audit is not None,'Missing base-load provenance')
 output.mkdir(parents=True)
 model.save_pretrained(str(output),safe_serialization=True,save_embedding_layers=False)
 model.config.save_pretrained(output/'text_config')
 if tokenizer is not None:tokenizer.save_pretrained(output/'tokenizer')
 from safetensors import safe_open
 with safe_open(str(output/'adapter_model.safetensors'),framework='pt',device='cpu') as f:
  tensors={name:{'shape':list(f.get_slice(name).get_shape()),'dtype':str(f.get_slice(name).get_dtype())} for name in f.keys()}
 score=[name for name in tensors if name.endswith(('score.weight','score.bias'))]
 require(len(score)==2,'Saved adapter omits the fully trained score head')
 require(all('lora_' in name or name in score for name in tensors),'Full pretrained weights leaked into adapter')
 cfg=json.loads((output/'adapter_config.json').read_text())
 require('score' in cfg.get('modules_to_save',[]),'PEFT config omits score modules_to_save')
 manifest={'format_version':1,'status':'complete','family':audit['family'],'label_order':list(BIO_LABELS),
  'base_identity':base_identity(audit),'lora':audit['lora'],'adapter_tensors':tensors,'score_keys':score,
  'metadata':metadata or {},'dependencies':audit['dependencies'],
  'files':{str(p.relative_to(output)):{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in sorted(output.rglob('*')) if p.is_file()}}
 write_json(output/'ner-adapter.json',manifest)
 return manifest


if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description='Header/meta audit only; never loads full weight tensors.')
 parser.add_argument('--model-key',choices=tuple(MODEL_SPECS),required=True)
 parser.add_argument('--source',type=Path,required=True)
 parser.add_argument('--output',type=Path,required=True)
 arguments=parser.parse_args()
 require(not arguments.output.exists(),'Preserve previous audit; choose a new output path')
 report=meta_shape_audit(arguments.model_key,arguments.source)
 write_json(arguments.output,report)
 print(json.dumps({key:report[key] for key in ('status','family','text_parameters','matched_text_tensors','excluded_tensors','model_weights_loaded')}))
 if report['status']!='pass':raise SystemExit(1)
