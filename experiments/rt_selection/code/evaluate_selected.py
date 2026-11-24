from common import *
import argparse, subprocess, shutil, math, traceback
p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--gpu',type=int,choices=[4,5],required=True);a=p.parse_args();setup(a.gpu)
BASE=Path('@WORKSPACE@/native420_benchmark');OLD=Path('@WORKSPACE@/rt_endpoint')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
try:
 cfg=read(ROOT/'protocol/experiment.json');assert all((ROOT/arm/'selection.json').exists() for arm in cfg['arms'])
 selection=read(ROOT/a.arm/'selection.json');selected=ROOT/a.arm/'selected.pth.tar';assert sha(selected)==selection['checkpoint_sha256']
 target=ROOT/'evaluation'/a.arm;target.mkdir(parents=True,exist_ok=True)
 old=OLD/'evaluation'/a.arm if a.arm!='surveillance' else Path('@WORKSPACE@/rt_matched/evaluation/surveillance')
 reuse=selection['step']==3000
 if reuse:
  import torch
  x=state(selected);y=state(OLD/a.arm/'endpoint.pth.tar');assert x.keys()==y.keys() and all(torch.equal(v,y[k]) for k,v in x.items());result_root=old
 else:
  result_root=target
  for d in ['code','protocol','inputs','references']:(target/d).mkdir(exist_ok=True)
  shutil.copy2(BASE/'code/common_yuv.py',target/'code/common_yuv.py')
  source=(BASE/'code/learned_worker.py').read_text();needle="w=OLD/'checkpoints'/f'cvpr2025_{name}.pth.tar';assert sha(w)==expected"
  replacement="w=Path("+repr(str(selected))+") if name=='video' else OLD/'checkpoints'/f'cvpr2025_{name}.pth.tar';assert sha(w)==("+repr(sha(selected))+" if name=='video' else expected)"
  assert source.count(needle)==1;source=source.replace(needle,replacement);worker=target/'code/learned_worker.py'
  if worker.exists():assert worker.read_text()==source
  else:worker.write_text(source)
  for n in ['input_manifest.json','source_sha256.json']:shutil.copy2(BASE/'protocol'/n,target/'protocol'/n)
  protocol=read(BASE/'protocol/benchmark.json');protocol.update(finetune_checkpoint_sha256=sha(selected),training_arm=a.arm,checkpoint_rule='best_full81_validation_saved_checkpoint',selected_step=selection['step'])
  if (target/'protocol/benchmark.json').exists():assert read(target/'protocol/benchmark.json')==protocol
  else:atomic(target/'protocol/benchmark.json',protocol)
  for n in ['rt_metrics.py','rt_transforms.py']:shutil.copy2(BASE/'references'/n,target/'references'/n)
  link=target/'inputs/yuv420'
  if not link.exists():link.symlink_to(BASE/'inputs/yuv420',target_is_directory=True)
  cmd=['@RT_PYTHON@','-B',str(worker),'--method','rt','--gpu',str(a.gpu),'--rates','0','1','2','3']
  with (target/'test.log').open('a') as f:
   subprocess.run(cmd+['--pilot'],cwd=target,stdout=f,stderr=subprocess.STDOUT,check=True)
   for q in range(4):assert read(target/f'results/rt/q{q}/cam0001_begin.json')['independent_decode_checked']
   atomic(target/'independent_decode_gate.json',dict(passed=True,points=4,frames=384,checkpoint_sha256=sha(selected)))
   subprocess.run(cmd,cwd=target,stdout=f,stderr=subprocess.STDOUT,check=True)
 count=0;undefined=[]
 for q in range(4):
  for c in read(BASE/'protocol/input_manifest.json')['clips']:
   r=read(result_root/f'results/rt/q{q}/{c["clip"]}.json');assert r['input_sha256']==c['sha256'] and r['true_bitstream'] and len(r['per_frame'])==96 and r['i_positions']==[0,32,64]
   assert all(math.isfinite(r[k]) for k in ['psnr','psnr_y','psnr_u','psnr_v','bpp'])
   assert all(sha(s['path'])==s['sha256'] for s in r['streams'])
   total=sum(Path(s['path']).stat().st_size for s in r['streams']);assert abs(r['bpp']-total*8/(640*384*96))<1e-10
   if not math.isfinite(r['ms_ssim']):undefined.append(dict(q=q,clip=c['clip'],metric='ms_ssim'))
   count+=1
 assert count==984
 atomic(target/'complete_audit.json',dict(passed=True,results=count,frames=count*96,selected_step=selection['step'],checkpoint_sha256=sha(selected),reused_prior_identical_endpoint=reuse,result_root=str(result_root),all_stream_hashes_verified=True,undefined_metrics=undefined,full82_all_metrics_valid=not undefined,test_used_for_selection=False))
except BaseException:atomic(ROOT/'evaluation'/a.arm/'failure.json',dict(error=traceback.format_exc(),no_checkpoint_fallback=True));raise
