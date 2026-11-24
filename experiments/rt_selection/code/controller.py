from common import *
setup()
import subprocess, fcntl, traceback, tarfile
lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
PY='@RT_PYTHON@';cfg=read(ROOT/'protocol/experiment.json');status=dict(status='preflight',pid=os.getpid(),jobs=[])
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def publish():status['updated']=time.strftime('%Y-%m-%d %H:%M:%S');atomic(ROOT/'controller_status.json',status)
def run_pair(items,phase):
 status.update(status=phase,jobs=[]);jobs=[]
 for script,args,label in items:
  f=(ROOT/'logs'/f'{label}.log').open('a');command=[PY,'-B',str(ROOT/'code'/script),*args];p=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  jobs.append((p,f,label));status['jobs'].append(dict(pid=p.pid,label=label,command=command))
 publish()
 while any(p.poll() is None for p,f,l in jobs):publish();time.sleep(20)
 for p,f,l in jobs:f.close();assert p.returncode==0,f'{l} failed; inspect retained logs, do not restart training'
 status['jobs']=[];publish()
try:
 (ROOT/'logs').mkdir(exist_ok=True)
 for n,h in cfg['code_sha256'].items():assert sha(ROOT/'code'/n)==h,n
 if not (ROOT/'protocol/input_checkpoint_audit.json').exists():run_pair([('preflight.py',[],'preflight')],'cpu_checkpoint_input_audit')
 assert read(ROOT/'protocol/input_checkpoint_audit.json')['passed']
 for pair in [[('vimeo',4),('surveillance',5)],[('shanghai',4),('ua_detrac',5)]]:
  tasks=[('validate_candidates.py',['--arm',a,'--gpu',str(g)],a+'_validation') for a,g in pair if not (ROOT/a/'selection.json').exists()]
  if tasks:run_pair(tasks,'all12_full81_validation')
 assert all(read(ROOT/a/'selection.json')['passed'] for a in cfg['arms'])
 atomic(ROOT/'protocol/all_selection_frozen_before_test.json',dict(passed=True,selections={a:read(ROOT/a/'selection.json') for a in cfg['arms']},test_used_for_selection=False))
 for pair in [[('vimeo',4),('surveillance',5)],[('shanghai',4),('ua_detrac',5)]]:
  tasks=[('evaluate_selected.py',['--arm',a,'--gpu',str(g)],a+'_test') for a,g in pair if not (ROOT/'evaluation'/a/'complete_audit.json').exists()]
  if tasks:run_pair(tasks,'selected_model_formal_test')
 audits={a:read(ROOT/'evaluation'/a/'complete_audit.json') for a in cfg['arms']};assert all(d['passed'] for d in audits.values())
 atomic(ROOT/'complete_audit.json',dict(passed=True,arms=audits,all_selections_independent_of_test=True,training_updates_added=0,full82_all_metrics_valid=all(r['full82_all_metrics_valid'] for r in audits.values())))
 status.update(status='server_audit_complete_archiving');publish();archive=ROOT/'best_validation_evidence.tar.gz';manifest=[]
 with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
  for directory,dirs,files in os.walk(ROOT,followlinks=False):
   dirs[:]=[d for d in dirs if d not in ['inputs','implementation','cache','tmp','__pycache__']]
   for name in files:
    p=Path(directory)/name
    if p==archive or p.is_symlink() or name.endswith(('.lock','.tar.gz','.partial')) or name in ['controller_status.json','archive_audit.json']:continue
    rel=p.relative_to(ROOT);manifest.append(dict(path=rel.as_posix(),sha256=sha(p),bytes=p.stat().st_size));tar.add(p,arcname=rel,recursive=False)
 atomic(ROOT/'archive_audit.json',dict(passed=True,sha256=sha(archive),bytes=archive.stat().st_size,files=manifest,artifact=str(archive)))
 status.update(status='server_complete_ready_for_local_sync',archive=str(archive));publish();print('SERVER_COMPLETE',flush=True)
except BaseException:status.update(status='failed',error=traceback.format_exc());publish();raise
