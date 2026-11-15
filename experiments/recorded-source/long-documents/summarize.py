"""Recount saved long-context answers; never loads a model or calls a service."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

LABELS=('PRIVATE_PRICE','NEGOTIATION_LIMIT','INDIVIDUAL_DISCOUNT',
        'RETRO_BONUS','PAYMENT_TERMS','INTERNAL_MARGIN')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8388608),b''):h.update(block)
    return h.hexdigest()


class Sources:
    def __init__(self):self.files={}
    def bytes(self,path):
        path=Path(path).resolve();raw=path.read_bytes()
        info={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        require(self.files.setdefault(str(path),info)==info,'Source changed during reading')
        return raw
    def read(self,path):return json.loads(self.bytes(path))
    def optional(self,path):return self.read(path) if Path(path).is_file() else None
    def verify(self):
        for p,v in self.files.items():
            require(Path(p).stat().st_size==v['bytes'] and sha(p)==v['sha256'],'Source changed before report: '+p)


def spans(text,values,scored=False):
    require(isinstance(values,list),'Spans must be a list')
    last=0
    for item in sorted(values,key=lambda x:(x['start'],x['end'])):
        a,b=item['start'],item['end']
        require(type(a) is int and type(b) is int and 0<=a<b<=len(text) and a>=last,'Invalid/overlapping offsets')
        require(item.get('text')==text[a:b] and item.get('label') in LABELS,'Span source text/type differs')
        if scored:
            score=item.get('score')
            require(type(score) in (int,float) and math.isfinite(score) and 0<=score<=1,'Invalid score')
        last=b


def fraction(a,b):return a/b if b else None


def prf(tp,fp,fn):
    return {'tp':tp,'fp':fp,'fn':fn,'precision':fraction(tp,tp+fp),
            'recall':fraction(tp,tp+fn),'f1':fraction(2*tp,2*tp+fp+fn)}


def covered_length(a,b,predicted):
    # Predicted intervals are already validated as disjoint. Avoid allocating a
    # character set proportional to document length for every long document.
    return sum(max(0,min(b,p['end'])-max(a,p['start'])) for p in predicted)


def metrics(rows,answers):
    require({r['id'] for r in rows}==set(answers),'Quality requires every selected document')
    totals=Counter();counts={k:Counter() for k in LABELS}
    for row in rows:
        answer=answers[row['id']];text=row['text'];gold=row['spans'];pred=answer['predictions']
        require(answer['text']==text,'Prediction text differs');spans(text,gold);spans(text,pred,True)
        g={(s['label'],s['start'],s['end']) for s in gold};p={(s['label'],s['start'],s['end']) for s in pred}
        for label in LABELS:
            gl={s for s in g if s[0]==label};pl={s for s in p if s[0]==label}
            counts[label].update(tp=len(gl&pl),fp=len(pl-gl),fn=len(gl-pl))
        totals.update(documents=1,gold_entities=len(gold),negative_documents=int(not gold),
                      negative_documents_masked=int(not gold and bool(pred)))
        for entity in gold:
            length=entity['end']-entity['start'];covered=covered_length(entity['start'],entity['end'],pred)
            totals.update(gold_characters=length,masked_gold_characters=covered,
                          fully_masked_entities=int(covered==length),
                          partially_masked_entities=int(0<covered<length),unmasked_entities=int(covered==0))
        target=row.get('contrast_target')
        if row.get('contrast_role')=='public' and target:
            a,b=target['start'],target['end']
            require(0<=a<b<=len(text) and text[a:b]==target['text'],'Invalid public target')
            totals.update(public_occurrences=1,public_masked=int(covered_length(a,b,pred)>0))
    micro={key:sum(c[key] for c in counts.values()) for key in ('tp','fp','fn')}
    return {'documents':len(rows),'micro':prf(**micro),
            'per_type':{k:prf(**{v:c[v] for v in ('tp','fp','fn')}) for k,c in counts.items()},
            'coverage':{k:totals[k] for k in ('gold_entities','fully_masked_entities','partially_masked_entities','unmasked_entities','gold_characters','masked_gold_characters')},
            'public':{'count':totals['public_occurrences'],'masked':totals['public_masked']},
            'negative_documents':totals['negative_documents'],'negative_documents_masked':totals['negative_documents_masked']}


def quantile(values,q):
    require(bool(values),'No measured values');values=sorted(values);position=(len(values)-1)*q
    a=int(math.floor(position));b=int(math.ceil(position));return values[a]+(values[b]-values[a])*(position-a)


def speed(answers,expected):
    if not answers:return None
    times=[]
    for r in answers:
        seconds=r.get('seconds')
        require(type(seconds) in (int,float) and math.isfinite(seconds) and seconds>0,'Invalid measured latency')
        times.append(seconds)
    def peak(field):
        found=[r[field] for r in answers if r.get(field) is not None]
        require(all(type(x) is int and x>=0 for x in found),'Invalid peak memory')
        return max(found) if found else None
    return {'status':'complete' if len(answers)==expected else 'partial',
            'documents':len(answers),'expected_documents':expected,'batch_size':1,
            'latency_p50_ms':quantile(times,.5)*1000,'latency_p95_ms':quantile(times,.95)*1000,
            'peak_allocated_bytes':peak('peak_allocated_bytes'),'peak_reserved_bytes':peak('peak_reserved_bytes'),
            'throughput':'not measured; never inferred from latency'}


def tier(row):return str(row['length_tier'])


def position(row):return str(row.get('position','none'))


def subset_result(rows,answers,issues=None):
    found={r['id']:answers[r['id']] for r in rows if r['id'] in answers}
    missing=[r['id'] for r in rows if r['id'] not in found]
    reasons=Counter((issues or {}).get(identifier,'not_started') for identifier in missing)
    if not missing:status='complete'
    elif set(reasons)=={'unsupported_length'}:status='unsupported_length'
    elif not found and len(reasons)==1:status=next(iter(reasons))
    else:status='partial'
    return {'status':status,'expected_documents':len(rows),'completed_documents':len(found),
            'missing_reasons':dict(reasons),
            'metrics':metrics(rows,found) if not missing else None,'speed':speed(list(found.values()),len(rows))}


def safe_relative(value):
    path=Path(value)
    require(not path.is_absolute() and '..' not in path.parts and bool(path.parts),'Unsafe artifact path')
    return path


def unsupported_records(sources,directory,rows,freeze,marker,support):
    """An unsupported tier needs a genuine OOM record, not only a status string."""
    unsupported={}
    for document in (support,marker):
        if not document:continue
        values=document.get('unsupported_tiers',{})
        require(isinstance(values,dict),'Unsupported tiers must be an object')
        for key,reason in values.items():
            key=str(key);require(key in {tier(r) for r in rows},'Unknown unsupported tier')
            require(key not in unsupported or unsupported[key]==reason,'Conflicting unsupported reasons')
            unsupported[key]=reason
    if support and freeze:
        for name in ('model_key','mode','runner_config_sha256'):
            if name in freeze and name in support:require(support[name]==freeze[name],'Support belongs to different configuration')
    for length,reason in unsupported.items():
        require(isinstance(reason,dict),'Unsupported reason must be an object')
        kind=reason.get('reason');require(kind in ('pilot_oom','warmup_oom','main_oom'),'Unknown unsupported reason')
        origin=directory.parent.parent/'pilot-001'/directory.name if kind=='pilot_oom' else directory
        if kind=='pilot_oom':
            pilot=sources.read(origin/'complete.json')
            require(pilot.get('pilot_only') is True and pilot.get('status') in ('complete','complete_with_unsupported'),'Invalid OOM pilot')
            for name in ('model_key','mode','runner_config_sha256'):
                if freeze and name in freeze:require(pilot.get(name)==freeze[name],'OOM pilot belongs to different configuration')
        path=origin/safe_relative(reason['record']);record=sources.read(path)
        require(record.get('status')=='oom' and record.get('predictions') is None,'Unsupported length lacks a genuine OOM')
        require(record.get('id')==reason.get('source_id') and str(record.get('length_tier'))==length,'OOM source/tier differs')
    return unsupported


def session_records(sources,runs):
    """Read actual main admission decisions, without assuming absent means OOM."""
    main=sources.optional(Path(runs)/'session.json')
    parent=sources.optional(Path(runs).parent/'session.json')
    values=main.get('stages',[]) if main is not None else parent.get('main_configurations',[]) if parent else []
    require(isinstance(values,list),'Session stages must be a list');result={}
    for item in values:
        key=item.get('id');require(isinstance(key,str) and key,'Session stage has no ID')
        require(key not in result,'Duplicate main session stage');result[key]=item
    return {'main':main,'parent':parent},result


def run_record(sources,directory,rows,data_sha,session_stage=None):
    stage=session_stage or {};stage_status=stage.get('status')
    not_started=stage_status if isinstance(stage_status,str) and stage_status.startswith('not_started_') else None
    if stage.get('expected_documents') is not None:require(stage['expected_documents']==len(rows),'Session cohort dimensions differ')
    require(not not_started or not directory.is_dir() or not any(directory.iterdir()),'Session says not started despite run artifacts')
    result={'directory':str(directory),'status':not_started or 'not_started','completed_documents':0,
            'attempted_documents':0,'skipped_documents':0,'expected_documents':len(rows),'metrics':None,
            'by_length':{},'by_length_position':{},'session_stage':session_stage}
    freeze=config=marker=skipped=failure=support=None
    answers={};attempts={};by_id={r['id']:r for r in rows}
    if directory.is_dir():
        freeze=sources.optional(directory/'freeze.json');config=sources.optional(directory/'config.json')
        marker=sources.optional(directory/'complete.json');skipped=sources.optional(directory/'skipped.json')
        failure=sources.optional(directory/'failure.json');support=sources.optional(directory/'support.json')
        if freeze:
            require(freeze.get('data_sha256')==data_sha,'Run uses a different dataset')
            require(freeze.get('documents')==len(rows),'Run data dimensions differ')
        for path in sorted((directory/'records').glob('*.json')):
            require(path.stem.isdigit(),'Record filename must be an original index')
            index=int(path.stem);require(0<=index<len(rows),'Record index out of bounds')
            value=sources.read(path);identifier=value.get('id')
            require(identifier in by_id,'Attempt ID absent from corpus')
            require(identifier not in attempts,'Duplicate answer/attempt ID')
            require(identifier==rows[index]['id'] and value.get('index',index)==index,'Record index/ID differs')
            require(value.get('text')==by_id[identifier]['text'],'Attempt original text differs')
            require(value.get('length_tier',by_id[identifier]['length_tier'])==by_id[identifier]['length_tier'],'Attempt length tier differs')
            require(value.get('status') in ('complete','oom','error'),'Unknown attempt status')
            attempts[identifier]=value
            if value['status']=='complete':
                require(not value.get('deadline_interrupted') and value.get('error',{}).get('type')!='DeadlineInterrupted','Interrupted answer marked complete')
                spans(value['text'],value['predictions'],True);answers[identifier]=value
            else:require(value.get('predictions') is None,'Failed attempt must not masquerade as an empty answer')
    if attempts:require(freeze is not None,'Answered run lacks data freeze')
    require(not not_started or not attempts and not marker and not support and not skipped,'Session says not started despite run artifacts')
    unsupported=unsupported_records(sources,directory,rows,freeze,marker,support)
    skip_by_id={}
    if skipped is not None:
        require(isinstance(skipped,list),'Skipped records must be a list')
        for value in skipped:
            identifier=value.get('id');index=value.get('index')
            require(identifier in by_id,'Skipped ID absent from corpus')
            require(identifier not in skip_by_id,'Duplicate skipped ID')
            require(identifier not in attempts,'Attempt and skipped IDs overlap')
            require(type(index) is int and 0<=index<len(rows) and rows[index]['id']==identifier,'Skipped index/ID differs')
            require(value.get('length_tier')==by_id[identifier]['length_tier'],'Skipped tier differs')
            require(value.get('status') in ('unsupported_length','not_started_deadline'),'Unknown skip status')
            if value['status']=='unsupported_length':
                require(tier(by_id[identifier]) in unsupported and value.get('reason')==unsupported[tier(by_id[identifier])],'Skipped length lacks matching OOM support')
            skip_by_id[identifier]=value
    if marker:
        if marker.get('status')=='complete':
            require(len(answers)==len(rows) and not skip_by_id,'Complete marker contradicts answers')
        for key,count in [('expected_documents',len(rows)),('attempted_documents',len(attempts)),
                          ('completed_documents',len(answers)),('skipped_documents',len(skip_by_id)),
                          ('oom_documents',sum(v['status']=='oom' for v in attempts.values()))]:
            require(type(marker.get(key)) is int and marker[key]==count,'Completion accounting differs: '+key)
        require(marker.get('quality_complete') is (len(answers)==len(rows)),'Completion quality flag contradicts answers')
        require(marker.get('data_sha256')==data_sha,'Completion data SHA differs')
        require(marker.get('status') in ('complete','partial','complete_with_unsupported'),'Unknown completion marker status')
        if marker['status']=='complete_with_unsupported':
            require(bool(unsupported) and (set(attempts)|set(skip_by_id))==set(by_id),'Unsupported completion lacks full accounting')
            require(all(v['status'] in ('complete','oom') for v in attempts.values()) and all(v['status']=='unsupported_length' for v in skip_by_id.values()),'Unsupported completion includes other failures')
        if stage.get('completion') is not None:require(stage['completion']==marker,'Session completion differs from artifact')
        if stage_status in ('complete','partial','complete_with_unsupported'):require(stage_status==marker['status'],'Session status differs from completion marker')
    if not marker:
        require(stage_status not in ('complete','complete_with_unsupported') and not stage.get('completion'),'Session claims completion without its saved marker')
    if len(answers)==len(rows):require(marker is not None and marker.get('status')=='complete','All answers require a final completion marker')
    status=marker['status'] if marker else 'partial' if attempts or skip_by_id else 'failed' if failure or stage_status=='failed' else not_started or stage_status or 'not_started'
    issues={}
    for row in rows:
        identifier=row['id']
        if identifier in answers:continue
        attempt=attempts.get(identifier);skip=skip_by_id.get(identifier)
        if attempt:
            issues[identifier]='unsupported_length' if attempt['status']=='oom' else 'deadline_interrupted' if attempt.get('error',{}).get('type')=='DeadlineInterrupted' else 'error'
        elif skip:issues[identifier]=skip['status']
        elif not_started:issues[identifier]=not_started
        elif tier(row) in unsupported:issues[identifier]='unsupported_length'
        elif stage_status=='deadline':issues[identifier]='not_started_deadline'
        elif failure or stage_status=='failed':issues[identifier]='not_started_after_failure'
        else:issues[identifier]='not_started'
    accounting=Counter('complete' if r['id'] in answers else issues[r['id']] for r in rows)
    result.update(status=status,config=config,freeze=freeze,completion=marker,skipped=skipped,failure=failure,support=support,
                  unsupported_tiers=unsupported,accounting=dict(accounting),completed_documents=len(answers),
                  attempted_documents=len(attempts),skipped_documents=len(skip_by_id),
                  metrics=metrics(rows,answers) if status=='complete' else None,
                  speed=speed(list(answers.values()),len(rows)),
                  error_attempts=[{k:r.get(k) for k in ('id','status','error','seconds','input_tokens','peak_allocated_bytes','peak_reserved_bytes')} for r in attempts.values() if r['status']!='complete'],
                  diagnostics={'decoder_error_documents':sum(bool(r.get('decoder_errors')) for r in answers.values()),
                               'truncated_documents':sum(bool(r.get('truncated')) for r in answers.values()),
                               'output_tokens':sum(r.get('output_tokens') or 0 for r in answers.values())})
    for length in sorted({tier(r) for r in rows},key=lambda x:int(x)):
        selected=[r for r in rows if tier(r)==length];result['by_length'][length]=subset_result(selected,answers,issues)
        for where in sorted({position(r) for r in selected}):
            group=[r for r in selected if position(r)==where]
            result['by_length_position'][length+'/'+where]=subset_result(group,answers,issues)
    result['by_source_group']={name:subset_result(group,answers,issues) for name,group in (
        ('main',[r for r in rows if r.get('contrast_id')]),
        ('neutral',[r for r in rows if not r.get('contrast_id')])) if group}
    return result


def scope_summary(configurations,records,session):
    """Keep the original plan denominator visible after an explicit exclusion."""
    excluded=[r['id'] for r in records if r['status']=='not_started_user_scope']
    parent=session.get('parent') or {};declared=parent.get('authorized_excluded_configurations')
    if declared is not None:
        require(isinstance(declared,list) and len(declared)==len(set(declared)),'Invalid authorized scope exclusions')
        require(set(declared)==set(excluded),'Session authorized scope exclusions differ from main stages')
    count=len(configurations)-len(excluded)
    if parent.get('in_scope_configurations') is not None:
        require(parent['in_scope_configurations']==count,'Session in-scope count differs from main stages')
    return {'original_planned_configurations':len(configurations),'in_scope_configurations':count,
            'user_excluded_configurations':excluded,
            'in_scope_status_counts':dict(Counter(r['status'] for r in records if r['id'] not in excluded))}


def build_snapshot(data_path,runs,plan_path):
    sources=Sources();raw=sources.bytes(data_path);data_sha=hashlib.sha256(raw).hexdigest()
    rows=[json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    require(len(rows)==1000 and len({r['id'] for r in rows})==1000,'Expected complete unique test1000')
    for row in rows:spans(row['text'],row['spans'])
    require(sum(len(r['spans']) for r in rows)==480,'Expected 480 gold entities in this controlled corpus')
    require(Counter(r.get('contrast_role','neutral') for r in rows)==Counter(private=480,public=480,neutral=40),'Wrong private/public/neutral composition')
    require(Counter(tier(r) for r in rows)==Counter({'1000':250,'2000':250,'4000':250,'8000':250}),'Wrong length-tier composition')
    plan=sources.read(plan_path);result=[]
    session,stages=session_records(sources,runs)
    configurations=plan['configurations'];require(len({c['id'] for c in configurations})==len(configurations),'Duplicate configuration')
    require(len({c['result_dir'] for c in configurations})==len(configurations),'Duplicate result destination')
    require(set(stages)<={c['id'] for c in configurations},'Session has an unplanned main stage')
    for cfg in configurations:
        relative=Path(cfg['result_dir']);require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe run path')
        stage=stages.get(cfg['id'])
        if stage:
            for key in ('model','mode'):
                if stage.get(key) is not None:require(stage[key]==cfg[key],'Session stage configuration differs')
        record=run_record(sources,Path(runs)/relative,rows,data_sha,stages.get(cfg['id']));record.update(id=cfg['id'],name=cfg.get('name',cfg['id']),model=cfg['model'],mode=cfg['mode']);result.append(record)
    scope=scope_summary(configurations,result,session);sources.verify()
    return {'status':'verified_saved_artifacts','created_utc':datetime.now(timezone.utc).isoformat(),
            'generator_sha256':sha(__file__),'data_sha256':data_sha,'documents':len(rows),'plan':plan,'session':session,'scope':scope,
            'records':result,'sources':sources.files,'verification':{'source_files':len(sources.files),'model_calls':0,'network_calls':0},
            'limitations':['Контролируемая синтетика; длины и позиции внутри одного семейства зависимы.',
                           'Reference-длина задается Qwen tokenizer; фактическая длина других моделей записана отдельно.',
                           'Сущность и пояснение статуса расположены локально; дальние смысловые зависимости не проверены.',
                           'Срез с OOM/пропусками не получает нулевого F1 и не называется полным test.',
                           'Это один проход batch1; throughput не выводится из latency.']}


def fmt(value,digits=2):return f'{value:.{digits}f}' if value is not None else '—'


def markdown(snapshot):
    scope=snapshot['scope']
    lines=['# Длинные документы: сохраненные результаты','',
           f"Корпус: {snapshot['documents']} документов. Качество и latency рассчитаны по сохраненным ответам без повторного инференса.",'',
           f"Исходный план: {scope['original_planned_configurations']} конфигураций. Текущий объем: {scope['in_scope_configurations']}; исключены пользователем: {len(scope['user_excluded_configurations'])}.",'',
           '| Система | Длина reference | Статус | Текстов / ожидалось | Exact F1, % | Полностью закрыто | Задеты публичные | P50, мс | P95, мс | Peak allocated, GiB |',
           '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for record in snapshot['records']:
        for length,group in record['by_length'].items():
            m,s=group['metrics'],group['speed'];f1=m['micro']['f1'] if m else None
            coverage=m['coverage'] if m else None;public=m['public'] if m else None
            values=[record['name'].replace('|','\\|'),length,group['status'],f"{group['completed_documents']} / {group['expected_documents']}",
                    fmt(f1*100 if f1 is not None else None),f"{coverage['fully_masked_entities']}/{coverage['gold_entities']}" if coverage else '—',
                    f"{public['masked']}/{public['count']}" if public else '—',fmt(s['latency_p50_ms']) if s else '—',fmt(s['latency_p95_ms']) if s else '—',
                    fmt(s['peak_allocated_bytes']/2**30) if s and s.get('peak_allocated_bytes') is not None else '—']
            lines.append('| '+' | '.join(values)+' |')
    lines+=['','unsupported_length означает подтвержденный OOM этой длины; not_started_budget — конфигурация не запускалась по оценке времени; not_started_user_scope — основной прогон исключен пользователем. Все строки исходного плана сохранены. Эти статусы не означают нулевой F1.','',
            'Для неполных строк времена относятся только к завершенным документам; качество полного среза не вычисляется.','']
    lines+=['- '+x for x in snapshot['limitations']]
    lines+=['',f"Проверены SHA256 {snapshot['verification']['source_files']} источников. Источники и срезы по позициям — в snapshot.json.",'']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--runs',required=True);p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    snapshot=build_snapshot(a.data,a.runs,a.plan);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2,allow_nan=False)+'\n');(out/'comparison.md').write_text(markdown(snapshot))
    print(json.dumps({'status':snapshot['status'],'models':{r['id']:r['status'] for r in snapshot['records']},'source_files':len(snapshot['sources'])}))


if __name__=='__main__':main()
