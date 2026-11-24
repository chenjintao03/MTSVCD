from common import *
import argparse,subprocess,shutil,time,collections
p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--gpu',type=int,choices=[4,5],required=True);a=p.parse_args();setup(a.gpu)
BASE=Path('@WORKSPACE@/native420_benchmark');target=ROOT/'evaluation'/a.arm
target.mkdir(parents=True,exist_ok=True);selected=ROOT/a.arm/'selected.pth.tar';selection=read(ROOT/a.arm/'selection.json')
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
if not selection['accepted']:
 import torch
 x=state(selected);y=state(P);assert x.keys()==y.keys() and all(torch.equal(x[k],y[k]) for k in x)
 result_root=BASE
 atomic(target/'reuse_audit.json',dict(exact_model_tensor_identity=True,reason='No validation improvement accepted; selected original P model. Frozen I, source YUV420 windows, IP32, QPs and codec configuration identical to completed baseline.',source=str(BASE),results=984,not_a_finetuning_gain=True))
else:
 for d in ['code','protocol','inputs','references']:(target/d).mkdir(exist_ok=True)
 shutil.copy2(BASE/'code/common_yuv.py',target/'code/common_yuv.py')
 source=(BASE/'code/learned_worker.py').read_text()
 needle="w=OLD/'checkpoints'/f'cvpr2025_{name}.pth.tar';assert sha(w)==expected"
 replacement="w=Path("+repr(str(selected))+") if name=='video' else OLD/'checkpoints'/f'cvpr2025_{name}.pth.tar';assert sha(w)==("+repr(digest(selected))+" if name=='video' else expected)"
 assert source.count(needle)==1;source=source.replace(needle,replacement)
 (target/'code/learned_worker.py').write_text(source)
 for name in ['input_manifest.json','source_sha256.json']:shutil.copy2(BASE/'protocol'/name,target/'protocol'/name)
 cfg=read(BASE/'protocol/benchmark.json');cfg['finetune_checkpoint_sha256']=digest(selected);cfg['training_arm']=a.arm;atomic(target/'protocol/benchmark.json',cfg)
 for name in ['rt_metrics.py','rt_transforms.py']:shutil.copy2(BASE/'references'/name,target/'references'/name)
 link=target/'inputs/yuv420'
 if not link.exists():link.symlink_to(BASE/'inputs/yuv420',target_is_directory=True)
 cmd=['@RT_PYTHON@','-B',str(target/'code/learned_worker.py'),'--method','rt','--gpu',str(a.gpu),'--rates','0','1','2','3']
 with (target/'test.log').open('a') as f:
  subprocess.run(cmd+['--pilot'],cwd=target,stdout=f,stderr=subprocess.STDOUT,check=True)
  for q in range(4):
   pilot=read(target/f'results/rt/q{q}/cam0001_begin.json');assert pilot['independent_decode_checked'] and len(pilot['per_frame'])==96
  atomic(target/'trained_model_pilot_audit.json',dict(passed=True,checkpoint_sha256=digest(selected),results=4,frames=384,fresh_independent_decode=True))
  subprocess.run(cmd,cwd=target,stdout=f,stderr=subprocess.STDOUT,check=True)
 result_root=target
curves=[];baseline=[];count=0
for q in range(4):
 values=[];orig=[]
 for c in read(BASE/'protocol/input_manifest.json')['clips']:
  r=read(result_root/f'results/rt/q{q}/{c["clip"]}.json');b=read(BASE/f'results/rt/q{q}/{c["clip"]}.json')
  assert r['input_sha256']==b['input_sha256']==c['sha256'] and len(r['per_frame'])==96
  assert r['true_bitstream'] and r['i_positions']==[0,32,64]
  for stream in r['streams']:assert digest(stream['path'])==stream['sha256']
  values.append(r);orig.append(b);count+=1
 def avg(rows):
  by=collections.defaultdict(list)
  for row in rows:by[row['camera']].append(row)
  assert len(by)==82 and all(len(v)==3 for v in by.values())
  keys=['bpp','psnr','psnr_y','psnr_u','psnr_v','ms_ssim']
  return dict(rate=q,**{k:sum(sum(r[k] for r in group)/3 for group in by.values())/82 for k in keys})
 curves.append(avg(values));baseline.append(avg(orig))
atomic(target/'complete_audit.json',dict(passed=True,results=count,frames=count*96,bitstream_hashes_verified=True,selection_used_only_validation=True))
atomic(target/'rd_summary.json',dict(arm=a.arm,finetuning_accepted=selection['accepted'],curves=curves,original=baseline,aggregation='82 original videos equal weight; each mean of three96-frame windows',test_used_for_checkpoint_selection=False))
