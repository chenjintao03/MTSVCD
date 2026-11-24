"""Sequential matched quality pairs; waits for the complete RT server audit."""
from common import *
setup()
import subprocess,fcntl,traceback
lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
status=dict(status='waiting_for_rt',pid=os.getpid());(ROOT/'logs').mkdir(exist_ok=True)
def update(**kw):status.update(kw,updated=time.strftime('%Y-%m-%d %H:%M:%S'));atomic(ROOT/'controller_status.json',status)
def run(script,log,args):
    with (ROOT/'logs'/log).open('a') as f:subprocess.run([PY,'-B',str(ROOT/'code'/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def identity_audit(cfg):
    assert all(sha(ROOT/path)==value for path,value in cfg['source_sha256'].items()),'Frozen adaptation code changed'
    assert all(sha(ROOT/'protocol'/name)==value for name,value in cfg['input_protocol_sha256'].items()),'Frozen split or source metadata changed'
    original=read(ROOT/'protocol/official_source_identity.json')['original_sources']
    assert all(sha(path)==value for path,value in original.items()),'Original TCM inference source changed'
    src=Path('@REPO_ROOT@/Deep Learning-based/DCVC-TCM')
    assert all(sha(ROOT/'implementation'/Path(path).relative_to(src))==value for path,value in original.items()),'Isolated inference clone changed'
    for row in read(ROOT/'diagnostic/cpu_model_gate.json')['checks']:
        assert sha(row['i_path'])==row['i_sha256'] and sha(row['p_original_path'])==row['p_original_sha256']
def parallel(script,label,quality=None):
    jobs=[]
    for arm,gpu in [('surveillance',4),('vimeo',5)]:
        args=['--arm',arm,'--gpu',str(gpu)];folder=ROOT/arm
        if quality is not None:args+=['--quality',str(quality)];folder/=f'q{quality}'
        if script=='train_matched.py':
            if (folder/'estimated_selection.json').exists():assert read(folder/'status.json')['status']=='estimated_selection_complete';continue
            if (folder/'last.pth.tar').exists():args+=['--resume']
        elif script=='real_validation_gate.py' and (folder/'real_validation_gate.json').exists():assert read(folder/'real_validation_gate.json')['passed'];continue
        elif script=='natural_validation.py' and (folder/'natural_status.json').exists() and read(folder/'natural_status.json')['status']=='complete':continue
        elif script=='evaluate.py' and (ROOT/'evaluation'/arm/'complete_audit.json').exists():assert read(ROOT/'evaluation'/arm/'complete_audit.json')['passed'];continue
        handle=(ROOT/'logs'/f'{label}_{arm}.log').open('a');proc=subprocess.Popen([PY,'-B',str(ROOT/'code'/script),*args],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT);jobs.append((proc,handle,arm))
    update(status=label,quality=quality,jobs=[dict(pid=p.pid,arm=a) for p,f,a in jobs])
    failures=[]
    for proc,handle,arm in jobs:
        code=proc.wait();handle.close()
        if code:failures.append(dict(arm=arm,exit_code=code))
    assert not failures,failures
try:
    cfg=read(ROOT/'protocol/experiment.json');assert cfg['training_ready'];identity_audit(cfg)
    assert read(ROOT/'protocol/generated_worker_syntax_audit.json')['passed']
    assert read(ROOT/'protocol/formal_worker_syntax_audit.json')['passed']
    update()
    while not (RT_ROOT/'complete_audit.json').exists():time.sleep(30)
    assert read(RT_ROOT/'complete_audit.json')['passed']
    identity_audit(cfg)
    # No RT worker process may still occupy the user's GPUs after its completion marker.
    rows=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True).stdout.splitlines()
    assert not any(str(RT_ROOT/'code') in r and any(n in r for n in ['train_matched.py','evaluate.py','validation_stream_gate.py']) for r in rows)
    for q in [1,2,3,4]:
        path=ROOT/f'diagnostic/q{q}/gradient_gate.json'
        signature={n:sha(ROOT/'code'/n) for n in ['common.py','data.py','adapter.py','gradient_gate.py']}
        if not path.exists() or read(path)['status']!='passed' or read(path).get('source_signature')!=signature:update(status='gradient_and_real_stream_gate',quality=q);run('gradient_gate.py',f'gate_q{q}.log',['--gpu','4','--quality',str(q)])
        assert read(path)['status']=='passed'
    for q in [1,2,3,4]:
        parallel('train_matched.py',f'q{q}_matched_training',q)
        parallel('real_validation_gate.py',f'q{q}_real_validation',q)
        parallel('natural_validation.py',f'q{q}_natural_validation_final_model',q)
    parallel('evaluate.py','formal_test')
    identity_audit(cfg)
    assert all(read(ROOT/'evaluation'/arm/'complete_audit.json')['passed'] for arm in ['surveillance','vimeo'])
    atomic(ROOT/'complete_audit.json',dict(passed=True,arms=2,qualities_per_arm=4,updates_per_quality_per_arm=3000,test_results_per_arm=984,frames_per_arm=94464,test_used_for_selection=False,local_sync_pending=True))
    update(status='server_complete_local_sync_and_cross_dataset_pending',jobs=[])
except BaseException:update(status='failed',error=traceback.format_exc());raise
