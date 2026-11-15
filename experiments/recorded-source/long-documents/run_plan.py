"""Execute one prepared pilot/main plan sequentially; no rental or lifecycle API."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def commands(plan,config,output,phase,deadline,pilot_root=None):
    if phase not in ('pilot','main'):raise ValueError('Unknown phase')
    if phase=='main' and pilot_root is None:raise ValueError('Main plan needs a completed resource pilot directory')
    seen=set();items=[];source=Path(__file__).parent
    for item in plan['configurations']:
        key=item['result_dir'];relative=Path(key)
        if relative.is_absolute() or '..' in relative.parts or key in seen:raise ValueError('Unsafe/duplicate result path')
        seen.add(key)
        argv=[sys.executable,str(source/('run_pilot.py' if phase=='pilot' else 'run_model.py')),
              '--config',str(config),'--model',item['model'],'--mode',item['mode'],
              '--data',plan['pilot_data'] if phase=='pilot' else plan['data'],
              '--output',str(Path(output)/relative),'--deadline-unix',str(deadline)]
        if phase=='main':argv+=['--pilot-data',plan['pilot_data'],'--pilot-report',str(Path(pilot_root)/relative/'complete.json')]
        items.append({'id':item['id'],'output':str(Path(output)/relative),'argv':argv})
    return items


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--phase',choices=('pilot','main'),required=True);p.add_argument('--pilot-root')
    p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());output=Path(a.output).resolve()
    jobs=commands(plan,Path(a.config).resolve(),output,a.phase,a.deadline_unix,a.pilot_root)
    if a.dry_run:
        print(json.dumps({'status':'dry_run','phase':a.phase,'jobs':jobs,'model_calls':0,'network_calls':0}));return
    if a.deadline_unix<=time.time():raise TimeoutError('Plan deadline has already passed')
    output.mkdir(parents=True,exist_ok=False)
    state={'status':'running','phase':a.phase,'deadline_unix':a.deadline_unix,
           'lifecycle_control':'none; ending this process does not stop GPU billing','stages':[]}
    write(output/'session.json',state)
    for job in jobs:
        item={'id':job['id'],'argv':job['argv']};state['stages'].append(item)
        if time.time()>=a.deadline_unix-30:
            item['status']='not_started_deadline';write(output/'session.json',state);continue
        item['started_unix']=time.time();write(output/'session.json',state)
        with (output/(job['id']+'.log')).open('x') as log:
            child=subprocess.Popen(job['argv'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            try:child.wait(timeout=max(1,a.deadline_unix-time.time()+15))
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                item['deadline_terminated']=True
        item.update(exit_code=child.returncode,finished_unix=time.time())
        marker=Path(job['output'])/'complete.json'
        item['status']=json.loads(marker.read_text()).get('status','invalid_marker') if marker.is_file() else 'failed'
        write(output/'session.json',state)
    statuses=[x['status'] for x in state['stages']]
    state['status']='complete' if all(s=='complete' for s in statuses) else 'complete_with_unsupported' if all(s in ('complete','complete_with_unsupported') for s in statuses) else 'partial'
    state['finished_unix']=time.time();write(output/'session.json',state)
    print(json.dumps(state))


if __name__=='__main__':main()
