"""CPU preparation now; GPU work strictly after the complete matched TCM audit."""
from common import *
setup()
import subprocess,fcntl,traceback,collections
lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(ROOT/'logs').mkdir(exist_ok=True);status=dict(status='cpu_preflight',pid=os.getpid())
def update(**kw):status.update(kw,updated=time.strftime('%Y-%m-%d %H:%M:%S'));atomic(ROOT/'controller_status.json',status)
def pair(pilot):
    jobs=[];phase='pilot' if pilot else 'full'
    for gpu,rates in [(4,[0,2]),(5,[1,3])]:
        handle=(ROOT/'logs'/f'{phase}_gpu{gpu}.log').open('a');args=[PY,'-B',str(ROOT/'code/worker.py'),'--gpu',str(gpu),'--rates',*map(str,rates)]
        if pilot:args+=['--pilot']
        proc=subprocess.Popen(args,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT);jobs.append((proc,handle,gpu))
    update(status=phase,jobs=[dict(pid=p.pid,gpu=gpu) for p,f,gpu in jobs]);failed=[]
    for proc,handle,gpu in jobs:
        value=proc.wait();handle.close()
        if value:failed.append(dict(gpu=gpu,exit=value))
    assert not failed,failed
def audit(clips,pilot):
    count=frames=0;groups=collections.Counter()
    for q in [0,1,2,3]:
        for c in clips:
            d=read(ROOT/f'results/rt/q{q}/{c["clip"]}.json')
            assert d['protocol_sha256']==sha(ROOT/'protocol/experiment.json') and d['input_sha256']==c['sha256'] and d['source_sha256']==c['source_sha256'] and d['independent_decode_checked']
            assert d['frames']==len(d['per_frame'])==96 and d['i_positions']==[0,32,64] and d['width']==c['width'] and d['height']==c['height']
            assert d['dual_entropy_coders']==(c['width']*c['height']>1280*720) and sum(v['bytes'] for v in d['per_frame'])==d['bitstream_bytes']
            assert all(sha(v['path'])==v['sha256'] for v in d['streams'])
            if pilot:assert d['independent_metric_check']
            count+=1;frames+=96;groups[c['dataset']]+=1
    return dict(passed=True,results=count,frames=frames,groups=dict(groups),all_points_fresh_independent_decode=True,all_bitstream_hashes_verified=True,independent_metric_check_all=pilot,protocol_sha256=sha(ROOT/'protocol/experiment.json'))
try:
    update();source_identity()
    if not (ROOT/'diagnostic/cpu_input_gate.json').exists() or not read(ROOT/'diagnostic/cpu_input_gate.json')['passed']:
        with (ROOT/'logs/cpu_preflight.log').open('a') as f:subprocess.run([PY,'-B',str(ROOT/'code/preflight.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    assert read(ROOT/'diagnostic/cpu_input_gate.json')['passed']
    update(status='waiting_for_tcm',jobs=[])
    while not (TCM/'complete_audit.json').exists():time.sleep(30)
    assert read(TCM/'complete_audit.json')['passed']
    processes=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True).stdout.splitlines()
    assert not any(str(TCM/'code') in p and any(n in p for n in ['train_matched.py','natural_validation.py','real_validation_gate.py','evaluate.py','gradient_gate.py']) for p in processes),'Matched TCM GPU jobs still active'
    clips=read(ROOT/'protocol/input_manifest.json')['clips'];chosen={}
    for c in clips:chosen.setdefault(c['dataset'],c)
    if not (ROOT/'pilot_gate.json').exists():pair(True);atomic(ROOT/'pilot_gate.json',audit(list(chosen.values()),True))
    assert read(ROOT/'pilot_gate.json')['passed'] and read(ROOT/'pilot_gate.json')['results']==16
    pair(False);result=audit(clips,False);assert result['results']==192 and result['frames']==18432
    source_identity()
    for c in {c['source_path']:c for c in clips}.values():assert sha(c['source_path'])==c['source_sha256']
    result.update(source_data_unchanged=True,local_sync_pending=True,training_performed=False,paused_uvg_vtm_not_resumed=True)
    atomic(ROOT/'complete_audit.json',result);update(status='server_complete_sync_and_descriptive_curves_pending',jobs=[])
except BaseException:update(status='failed',error=traceback.format_exc());raise
