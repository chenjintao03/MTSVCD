from pathlib import Path
import os,json,time,hashlib,subprocess,fcntl,shutil,math,traceback,tarfile
R=Path(__file__).resolve().parents[1];os.chdir(R)
for k,v in dict(TEMP='tmp',TMP='tmp',TMPDIR='tmp',XDG_CACHE_HOME='cache').items():
 p=R/v;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
os.environ['PYTHONDONTWRITEBYTECODE']='1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False));t.replace(p)
def status(s,**kw):write(R/'controller_status.json',dict(status=s,pid=os.getpid(),time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw))
lock=(R/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
BASE=Path('@WORKSPACE@/native420_benchmark');DIAG=Path('@DATASET_ROOT@/rt_vimeo_regression_20261113');PY='@RT_PYTHON@'
try:
 status('preflight');cfg=read(R/'protocol/experiment.json')
 for n,h in cfg['code_sha256'].items():assert sha(R/'code'/n)==h
 for n in ['protocol','references','inputs','logs','weights']:(R/n).mkdir(exist_ok=True)
 assert read(DIAG/'complete_audit.json')['passed'] and sha(DIAG/'weights/step0600.pth.tar')==cfg['selected_p_sha256']
 for n in ['common_yuv.py','learned_worker.py']:assert sha(R/'code'/n)==sha(DIAG/'validation/step0600_monitor/code'/n)
 for n in ['input_manifest.json','source_sha256.json']:shutil.copy2(BASE/'protocol'/n,R/'protocol'/n)
 for n in ['rt_metrics.py','rt_transforms.py']:shutil.copy2(BASE/'references'/n,R/'references'/n)
 sources=read(R/'protocol/source_sha256.json')['rt'];assert all(sha(p)==h for p,h in sources.items())
 clips=read(R/'protocol/input_manifest.json')['clips'];assert len(clips)==246 and len({c['camera'] for c in clips})==82
 for c in clips:assert sha(BASE/'inputs/yuv420'/f'{c["clip"]}.yuv')==c['sha256']
 link=R/'inputs/yuv420'
 if not link.exists():link.symlink_to(BASE/'inputs/yuv420',target_is_directory=True)
 shutil.copy2(DIAG/'weights/step0600.pth.tar',R/'weights/selected_p.pth.tar');shutil.copy2(DIAG/'weights/step0600.pth.json',R/'weights/export_audit.json')
 benchmark=read(BASE/'protocol/benchmark.json');benchmark.update(selected_p_sha256=cfg['selected_p_sha256'],training_arm='vimeo_rerun_20261113',checkpoint_step=600,selection='explicit_user_choice_before_formal_test',validation_only=False)
 bp=R/'protocol/benchmark.json'
 if bp.exists():assert read(bp)==benchmark
 else:write(bp,benchmark)
 write(R/'protocol/preflight_audit.json',dict(passed=True,exact_worker_and_metric_sources_reused=True,all246_inputs_sha256_checked=True,selected_step=600,new_rerun_checkpoint=True,independent_decode_every_point=True,original_i_unchanged=True,physical_gpus=[4,5],outputs_cwd_cache_d_only=True))
 def run(pilot):
  jobs=[]
  for gpu,rates in [(4,[0,2]),(5,[1,3])]:
   cmd=[PY,'-B',str(R/'code/learned_worker.py'),'--method','rt','--gpu',str(gpu),'--rates',*map(str,rates)]+(['--pilot'] if pilot else [])
   f=(R/'logs'/f'{"pilot" if pilot else "full"}_gpu{gpu}.log').open('ab');p=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT);jobs.append((p,f,cmd))
  status('pilot' if pilot else 'formal_test',jobs=[dict(pid=p.pid,command=cmd) for p,f,cmd in jobs]);codes=[]
  for p,f,cmd in jobs:codes.append(p.wait());f.close()
  assert codes==[0,0],f'Inspect worker logs and preserve valid points: {codes}'
 run(True)
 assert all(read(R/f'results/rt/q{q}/{clips[0]["clip"]}.json')['independent_decode_checked'] for q in range(4))
 write(R/'protocol/pilot_gate.json',dict(passed=True,points=4,frames=384,new_i_p_independent_decode=True,checkpoint_sha256=cfg['selected_p_sha256']))
 run(False);status('auditing');points=[];fields=['bytes','psnr','psnr_y','psnr_u','psnr_v','ms_ssim']
 for q in range(4):
  for c in clips:
   d=read(R/f'results/rt/q{q}/{c["clip"]}.json');b=read(BASE/f'results/rt/q{q}/{c["clip"]}.json')
   assert d['input_sha256']==c['sha256'] and d['protocol_sha256']==sha(bp) and d['independent_decode_checked'] and d['true_bitstream']
   assert d['frames']==len(d['per_frame'])==96 and d['i_positions']==[0,32,64]
   assert all(math.isfinite(d[k]) for k in ['bpp','psnr','psnr_y','psnr_u','psnr_v','ms_ssim']) and 0<d['ms_ssim']<=1
   assert [{k:f[k] for k in fields} for f in d['per_frame'] if f['type']=='I']==[{k:f[k] for k in fields} for f in b['per_frame'] if f['type']=='I']
   size=0
   for s in d['streams']:assert sha(s['path'])==s['sha256'];size+=Path(s['path']).stat().st_size
   assert abs(d['bpp']-size*8/(640*384*96))<1e-12
   points.append(dict(camera=c['camera'],clip=c['clip'],quality=q,qp=[0,21,42,63][q],bpp=d['bpp'],psnr=d['psnr'],ms_ssim=d['ms_ssim']))
 assert len(points)==984 and all(sha(p)==h for p,h in sources.items()) and sha(DIAG/'weights/step0600.pth.tar')==cfg['selected_p_sha256']
 for n,h in cfg['code_sha256'].items():assert sha(R/'code'/n)==h
 write(R/'compact_results.json',dict(passed=True,checkpoint_step=600,selected_p_sha256=cfg['selected_p_sha256'],points=points))
 write(R/'complete_audit.json',dict(passed=True,results=984,frames=94464,videos=82,all_points_independent_decode=True,all_stream_sha256_verified=True,all_input_and_i_identity_verified=True,original_sources_unchanged=True,selection='explicit_user_choice_600_before_test',selected_p_sha256=cfg['selected_p_sha256']))
 status('archiving');files=[]
 for folder in ['code','protocol','references','results','bitstreams','logs','weights']:files.extend(p for p in (R/folder).rglob('*') if p.is_file())
 files.extend([R/'compact_results.json',R/'complete_audit.json']);entries=[dict(path=p.relative_to(R).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files];archive=R/'vimeo600_formal_evidence.tar.gz'
 with tarfile.open(archive,'w:gz',compresslevel=1) as t:
  for p in files:t.add(p,arcname=p.relative_to(R).as_posix())
 write(R/'archive_audit.json',dict(passed=True,files=entries,bytes=archive.stat().st_size,sha256=sha(archive)));status('complete');print('VIMEO600_FORMAL_COMPLETE',flush=True)
except BaseException:status('failed',error=traceback.format_exc());raise
