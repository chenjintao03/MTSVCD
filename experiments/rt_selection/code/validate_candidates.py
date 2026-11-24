from common import *
import argparse, math, fcntl, traceback
p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=['vimeo','shanghai','surveillance','ua_detrac']);p.add_argument('--gpu',required=True,type=int,choices=[4,5]);a=p.parse_args();setup(a.gpu)
import torch
from model import Video,IntraForTraining
from data import Validation,ThreadBatches
torch.set_num_threads(2);torch.manual_seed(20261104)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
cfg=read(ROOT/'protocol/experiment.json');audit=read(ROOT/'protocol/input_checkpoint_audit.json');assert audit['passed']
folder=ROOT/a.arm;folder.mkdir(exist_ok=True);lock=(folder/'validation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
status=dict(status='starting',pid=os.getpid(),arm=a.arm,gpu=a.gpu,completed_candidates=0)
def update(**kw):status.update(kw,updated=time.strftime('%Y-%m-%d %H:%M:%S'));atomic(folder/'status.json',status)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
signature=hashlib.sha256(json.dumps(dict(code=cfg['code_sha256'],inputs=audit['validation_inputs'],sources=audit['sources'],i=audit['original_i']),sort_keys=True).encode()).hexdigest()
def load(cls,path):
 net=cls().cuda();net.load_state_dict(state(path),strict=True);net.eval();net.requires_grad_(False);return net
try:
 assert torch.cuda.device_count()==1
 assert str(torch.cuda.get_device_properties(0).uuid).lower().replace('gpu-','')==os.environ['CUDA_VISIBLE_DEVICES'].lower().replace('gpu-','')
 intra=load(IntraForTraining,I);model=load(Video,P);dataset=Validation('surveillance');assert len(dataset)==243
 table=torch.exp(torch.linspace(0,math.log(768),64,device='cuda'))
 def validate(path,step,label,dest):
  checkpoint_sha=sha(path);model.load_state_dict(state(path),strict=True);model.eval();rows={}
  for qp in cfg['qps']:
   out=dest/f'qp{qp}.json';identity=dict(source_signature=signature,checkpoint_sha256=checkpoint_sha,qp=qp,step=step)
   if out.exists():
    r=read(out);assert r['identity']==identity and r['windows']==243;rows[str(qp)]=r['means'];continue
   update(status='full_validation',checkpoint_step=step,label=label,qp=qp,validation_done=0,validation_windows=243)
   sums={};per_window=[]
   with torch.no_grad():
    for index,frames in enumerate(ThreadBatches(dataset,1,2,2)):
     q=torch.tensor([qp],device='cuda');value=model(frames.cuda(),q,intra,table[q],True,False);values={k:float(v) for k,v in value.items()}
     if not all(math.isfinite(v) for v in values.values()):
      atomic(dest/'nonfinite.json',dict(identity=identity,window_index=index,metrics={k:str(v) for k,v in values.items()}));return None
     for k,v in values.items():sums[k]=sums.get(k,0)+v
     per_window.append(dict(index=index,**values))
     if (index+1)%6==0:update(validation_done=index+1)
   assert len(per_window)==243
   means={k:v/243 for k,v in sums.items()};atomic(out,dict(identity=identity,windows=243,means=means,per_window=per_window));rows[str(qp)]=means
  atomic(dest/'complete.json',dict(passed=True,step=step,metrics=rows,checkpoint_sha256=checkpoint_sha,source_signature=signature,windows=243,videos=81));return rows
 baseline_path=ROOT/'baseline/complete.json'
 if a.arm=='vimeo':assert validate(P,0,'original',ROOT/'baseline') is not None
 results=[]
 for item in audit['checkpoints'][a.arm]:
  assert sha(item['path'])==item['sha256']
  result=validate(item['path'],item['step'],'candidate',folder/f'validation/step{item["step"]:04d}')
  results.append(dict(step=item['step'],path=item['path'],checkpoint_sha256=item['sha256'],metrics=result,finite=result is not None));update(completed_candidates=len(results));print('CANDIDATE_COMPLETE',a.arm,item['step'],flush=True)
 while not baseline_path.exists():update(status='waiting_for_shared_original_baseline');time.sleep(10)
 b=read(baseline_path);assert b['source_signature']==signature;baseline=b['metrics'];valid=[]
 for r in results:
  r['normalized_score']=sum(r['metrics'][str(q)]['loss']/baseline[str(q)]['loss'] for q in cfg['qps'])/4 if r['finite'] else None
  if r['finite']:valid.append(r)
 assert valid,'No finite saved candidate; do not substitute original'
 best=min(valid,key=lambda r:(r['normalized_score'],r['step']));model.load_state_dict(state(best['path']),strict=True);selected=folder/'selected.pth.tar'
 if selected.exists():
  previous=state(selected);assert all(torch.equal(previous[k],v.cpu()) for k,v in model.state_dict().items())
 else:torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()}),selected)
 x=state(selected);y=state(best['path']);assert x.keys()==y.keys() and all(torch.equal(v,y[k]) for k,v in x.items())
 atomic(folder/'selection.json',dict(passed=True,arm=a.arm,step=best['step'],normalized_score=best['normalized_score'],checkpoint_sha256=sha(selected),candidate_sha256=best['checkpoint_sha256'],candidates=results,validation_videos=81,candidate_count=12,test_used_for_selection=False,original_fallback=False,source_signature=signature,estimated_rd_selection=True))
 update(status='selection_complete',selected_step=best['step'],normalized_score=best['normalized_score']);print('SELECTED',a.arm,best['step'],best['normalized_score'],flush=True)
except BaseException:update(status='failed',error=traceback.format_exc());raise
