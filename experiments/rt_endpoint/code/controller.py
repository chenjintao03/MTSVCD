"""Foreground fixed-endpoint experiment. No model selection or baseline fallback from test scores."""
from common import *
setup()
import subprocess,fcntl,traceback,urllib.request,http.client,tarfile,ast
lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
PY='@RT_PYTHON@';OLD=Path('@WORKSPACE@/rt_matched')
URL=read(ROOT/'protocol/transport.json')['base_url']
status=dict(status='preflight',pid=os.getpid(),jobs=[],fixed_endpoint=3000)
def post(path):
    b=Path(path).read_bytes();req=urllib.request.Request(URL+'/receive/'+Path(path).name,data=b,method='POST',headers={'X-Content-SHA256':hashlib.sha256(b).hexdigest()})
    with urllib.request.urlopen(req,timeout=30) as response:response.read()
def publish():
    status['updated']=time.strftime('%Y-%m-%d %H:%M:%S');snap=dict(status)
    snap['arms']={}
    for arm in ['vimeo','shanghai','ua_detrac']:
        row={}
        for name in ['status.json','endpoint_audit.json']:
            p=ROOT/arm/name
            if p.exists():row[name]=read(p)
        row['formal_results']=len(list((ROOT/'evaluation'/arm/'results/rt').glob('q*/*.json')))
        snap['arms'][arm]=row
    atomic(ROOT/'latest_status.json',snap)
    try:post(ROOT/'latest_status.json')
    except Exception as e:print('STATUS_TRANSPORT_RETRY',type(e).__name__,flush=True)
def run_pair(items,phase):
    status.update(status=phase,jobs=[]);jobs=[]
    for script,args,label in items:
        f=(ROOT/'logs'/f'{label}.log').open('a');p=subprocess.Popen([PY,'-B',str(ROOT/'code'/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        jobs.append((p,f,label));status['jobs'].append(dict(pid=p.pid,label=label,command=[script,*args]))
    while any(p.poll() is None for p,f,label in jobs):publish();time.sleep(30)
    for p,f,label in jobs:f.close();assert p.returncode==0,f'{label} failed; retained fixed checkpoints and evidence; inspect log'
    status['jobs']=[];publish()
def signature_check():
    cfg=read(ROOT/'protocol/experiment.json')
    for n,h in cfg['code_sha256'].items():assert hashlib.sha256((ROOT/'code'/n).read_bytes()).hexdigest()==h,n
try:
    (ROOT/'logs').mkdir(exist_ok=True);signature_check();publish()
    if not (ROOT/'diagnostic/input_audit.json').exists() or read(ROOT/'diagnostic/input_audit.json')['status']!='passed':run_pair([('preflight.py',[],'preflight')],'input_preflight')
    for name in ['cleanup_result.json','shanghai_copy_audit.json','diagnostic/input_audit.json','protocol/reused_endpoint_audit.json']:
        if (ROOT/name).exists():post(ROOT/name)
    todo=[]
    for arm,gpu in [('shanghai',4),('ua_detrac',5)]:
        if not (ROOT/f'diagnostic/{arm}_gradient_gate.json').exists() or read(ROOT/f'diagnostic/{arm}_gradient_gate.json')['status']!='passed':todo.append(('gradient_gate.py',['--gpu',str(gpu),'--arm',arm],arm+'_gradient_gate'))
    if todo:run_pair(todo,'external_input_real_stream_and_gradient_gates')
    assert all(read(ROOT/f'diagnostic/{a}_gradient_gate.json')['status']=='passed' for a in ['shanghai','ua_detrac'])
    todo=[]
    for arm,gpu in [('shanghai',4),('ua_detrac',5)]:
        if (ROOT/arm/'endpoint_audit.json').exists() and read(ROOT/arm/'endpoint_audit.json')['passed']:continue
        args=['--arm',arm,'--gpu',str(gpu)]
        if (ROOT/arm/'last.pth.tar').exists():args+=['--resume']
        todo.append(('train_endpoint.py',args,arm+'_training'))
    if todo:run_pair(todo,'fixed_3000_training')
    for pair in [[('vimeo',4),('shanghai',5)],[('ua_detrac',4)]]:
        todo=[]
        for arm,gpu in pair:
            if (ROOT/'evaluation'/arm/'complete_audit.json').exists() and read(ROOT/'evaluation'/arm/'complete_audit.json')['passed']:continue
            todo.append(('evaluate_endpoint.py',['--arm',arm,'--gpu',str(gpu)],arm+'_formal_test'))
        if todo:run_pair(todo,'fixed_endpoint_formal_test')
    assert all(read(ROOT/'evaluation'/a/'complete_audit.json')['passed'] for a in ['vimeo','shanghai','ua_detrac'])
    atomic(ROOT/'complete_audit.json',dict(passed=True,arms=['original','vimeo','shanghai','surveillance','ua_detrac'],new_tests=3,results_per_arm=984,frames_per_arm=94464,checkpoint_rule='fixed_update3000_no_performance_fallback',training_updates_per_arm=3000,reused_vimeo_actual_training=True,reused_surveillance_actual_training_and_test=True,test_used_for_selection=False))
    status.update(status='server_complete_collecting',jobs=[]);publish()
    archive=ROOT/'rt_dataset_endpoint_complete.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for folder in ['code','protocol','diagnostic','logs','evaluation','revisions']:
            for directory,dirs,files in os.walk(ROOT/folder,followlinks=False):
                dirs[:]=[d for d in dirs if d not in ['inputs','tmp','cache','__pycache__']]
                for name in files:
                    p=Path(directory)/name
                    if not p.is_symlink():tar.add(p,arcname=p.relative_to(ROOT),recursive=False)
        for arm in ['vimeo','shanghai','ua_detrac','surveillance']:
            for name in ['endpoint.pth.tar','endpoint_audit.json','status.json','metrics.jsonl','selection.json']:
                p=ROOT/arm/name
                if p.exists():tar.add(p,arcname=p.relative_to(ROOT),recursive=False)
        tar.add(ROOT/'complete_audit.json',arcname='complete_audit.json')
    # Stream transfer instead of loading an archive into RAM.
    from urllib.parse import urlparse
    u=urlparse(URL+'/receive/'+archive.name);h=hashlib.sha256()
    with archive.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    conn=http.client.HTTPConnection(u.hostname,u.port,timeout=120);conn.putrequest('POST',u.path);conn.putheader('Content-Length',str(archive.stat().st_size));conn.putheader('X-Content-SHA256',h.hexdigest());conn.endheaders()
    with archive.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):conn.send(b)
    response=conn.getresponse();assert response.status==200;response.read();conn.close()
    atomic(ROOT/'collection_audit.json',dict(passed=True,archive_sha256=h.hexdigest(),bytes=archive.stat().st_size,all_new_artifacts_sent_to_local_D=True,server_last_and_candidates_retained=True));post(ROOT/'collection_audit.json')
    status.update(status='complete_local_archive_delivered');publish()
except BaseException:
    status.update(status='failed',error=traceback.format_exc(),no_checkpoint_fallback=True);publish();raise
