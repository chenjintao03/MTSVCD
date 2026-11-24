from common import *
setup()
import subprocess,fcntl,traceback
lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
PY='@RT_PYTHON@'
status=dict(status='preflight',pid=os.getpid())
def update(**kw):status.update(kw,updated=time.strftime('%Y-%m-%d %H:%M:%S'));atomic(ROOT/'controller_status.json',status)
def run(script,log,args=[]):
    with (ROOT/'logs'/log).open('a') as f:subprocess.run([PY,'-B',str(ROOT/'code'/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
    (ROOT/'logs').mkdir(exist_ok=True);update()
    if not (ROOT/'diagnostic/input_audit.json').exists() or read(ROOT/'diagnostic/input_audit.json')['status']!='passed':run('preflight.py','preflight.log')
    update(status='gradient_and_real_stream_gate')
    if not (ROOT/'diagnostic/gradient_gate.json').exists() or read(ROOT/'diagnostic/gradient_gate.json')['status']!='passed':run('gradient_gate.py','gradient_gate.log',['--gpu','4'])
    assert read(ROOT/'diagnostic/gradient_gate.json')['status']=='passed';jobs=[]
    for arm,gpu in [('surveillance',4),('vimeo',5)]:
        if (ROOT/arm/'status.json').exists() and read(ROOT/arm/'status.json')['status']=='finetuning_complete':continue
        f=(ROOT/'logs'/f'{arm}.log').open('a');p=subprocess.Popen([PY,'-B',str(ROOT/'code/train_matched.py'),'--arm',arm,'--gpu',str(gpu)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);jobs.append((p,f,arm))
    update(status='matched_finetuning',jobs=[dict(pid=p.pid,arm=a) for p,f,a in jobs])
    for p,f,arm in jobs:assert p.wait()==0,f'{arm} failed';f.close()
    assert all(read(ROOT/a/'selection.json')['updates_attempted']==3000 for a in ['surveillance','vimeo'])
    update(status='real_validation_guard');jobs=[]
    for arm,gpu in [('surveillance',4),('vimeo',5)]:
        if (ROOT/arm/'real_validation_gate.json').exists() and read(ROOT/arm/'real_validation_gate.json')['passed']:continue
        f=(ROOT/'logs'/f'{arm}_real_validation.log').open('a');p=subprocess.Popen([PY,'-B',str(ROOT/'code/validation_stream_gate.py'),'--arm',arm,'--gpu',str(gpu)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);jobs.append((p,f,arm))
    for p,f,arm in jobs:assert p.wait()==0,f'{arm} real validation guard failed';f.close()
    update(status='formal_test');jobs=[]
    for arm,gpu in [('surveillance',4),('vimeo',5)]:
        if (ROOT/'evaluation'/arm/'complete_audit.json').exists() and read(ROOT/'evaluation'/arm/'complete_audit.json')['passed']:continue
        f=(ROOT/'logs'/f'{arm}_test.log').open('a');p=subprocess.Popen([PY,'-B',str(ROOT/'code/evaluate.py'),'--arm',arm,'--gpu',str(gpu)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);jobs.append((p,f,arm))
    for p,f,arm in jobs:assert p.wait()==0,f'{arm} formal test failed';f.close()
    atomic(ROOT/'complete_audit.json',dict(passed=True,arms=2,updates_per_arm=3000,test_results_per_arm=984,frames_per_arm=94464,test_not_used_for_selection=True));update(status='server_complete_sync_and_TCM_review_pending',jobs=[])
except BaseException:update(status='failed',error=traceback.format_exc());raise
