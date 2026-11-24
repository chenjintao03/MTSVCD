from common import *
import argparse,random,math,traceback,fcntl,copy
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['shanghai','ua_detrac'],required=True);p.add_argument('--gpu',type=int,choices=[4,5],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();setup(a.gpu)
import torch
import numpy as np
from model import Video,IntraForTraining
from data import Clips,Validation,ThreadBatches
torch.set_num_threads(2);torch.manual_seed(20261104);random.seed(20261104);np.random.seed(20261104)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
folder=ROOT/a.arm;folder.mkdir(exist_ok=True);lock=(folder/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
cfg=read(ROOT/'protocol/experiment.json');status=dict(status='starting',pid=os.getpid(),arm=a.arm,gpu=a.gpu,updates=0,target_updates=3000)
def update(**kw):status.update(kw,updated=time.strftime('%Y-%m-%d %H:%M:%S'));atomic(folder/'status.json',status)
def log(event,**kw):
    with (folder/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(event=event,time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw))+'\n')
def load(cls,path):
    m=cls().cuda();m.load_state_dict(state(path),strict=True);return m
def checkpoint(path,step):
    tmp=path.with_suffix('.tmp');torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),updates=step,protocol=cfg,python_random=random.getstate(),numpy_random=np.random.get_state(),torch_random=torch.get_rng_state(),cuda_random=torch.cuda.get_rng_state_all(),qp_random=qp_rng.getstate(),sampler_offset=step*16),tmp);tmp.replace(path)
def validate(dataset,label,step):
    model.eval();result={};update(status='validating',validation=label,updates=step,validation_clips=len(dataset))
    with torch.no_grad():
        for qp in [0,21,42,63]:
            sums={};count=0
            for frames in ThreadBatches(dataset,1,2,2):
                q=torch.tensor([qp],device='cuda');r=model(frames.cuda(),q,intra,table[q],True,False)
                for k,v in r.items():sums[k]=sums.get(k,0)+float(v)
                count+=1
                if count%30==0:update(validation_qp=qp,validation_done=count)
            result[str(qp)]={k:v/count for k,v in sums.items()}
    model.train();log('validation',step=step,label=label,rates=result);return result
def score(candidate,base):return sum(candidate[q]['loss']/base[q]['loss'] for q in base)/4
def eligible(candidate,base):return score(candidate,base)<1 and max(candidate[q]['loss']/base[q]['loss']-1 for q in base)<=.02
class Offset:
    def __init__(self,ds,offset):self.ds=ds;self.offset=offset
    def __len__(self):return len(self.ds)-self.offset
    def __getitem__(self,n):return self.ds[n+self.offset]
try:
    assert read(ROOT/f'diagnostic/{a.arm}_gradient_gate.json')['status']=='passed'
    assert str(torch.cuda.get_device_properties(0).uuid).lower().replace('gpu-','')==os.environ['CUDA_VISIBLE_DEVICES'].lower().replace('gpu-','')
    intra=load(IntraForTraining,I).eval();intra.requires_grad_(False);model=load(Video,P)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-6);qp_rng=random.Random(20261104);table=torch.exp(torch.linspace(0,math.log(768),64,device='cuda'))
    start=0;last=folder/'last.pth.tar';assert a.resume or not last.exists(),'Explicit audited resume required'
    if last.exists():
        ckpt=torch.load(last,map_location='cpu',weights_only=False);assert ckpt['protocol']==cfg and ckpt['sampler_offset']==ckpt['updates']*16
        model.load_state_dict(ckpt['state_dict']);optimizer.load_state_dict(ckpt['optimizer']);start=ckpt['updates'];random.setstate(ckpt['python_random']);np.random.set_state(ckpt['numpy_random']);torch.set_rng_state(ckpt['torch_random']);torch.cuda.set_rng_state_all(ckpt['cuda_random']);qp_rng.setstate(ckpt['qp_random']);log('explicit_state_resume',step=start)
    screen=Validation('surveillance',limit=8);full=Validation('surveillance')
    if (folder/'baseline_screen.json').exists():base=read(folder/'baseline_screen.json')
    else:
        assert start==0;base=validate(screen,'common_surveillance_baseline',0);atomic(folder/'baseline_screen.json',base)
    ds=Clips(a.arm,'train',steps=3000,effective_batch=16,seed=20261104);iterator=iter(ThreadBatches(Offset(ds,start*16),4,2,4))
    for step in range(start+1,3001):
        lr=(1e-6+(5e-6-1e-6)*min(step/100,1))*(.5 if step>1500 else 1)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True);totals={};qps=[qp_rng.randrange(16)+16*(n%4) for n in range(16)];qp_rng.shuffle(qps)
        for offset in range(0,16,4):
            x=next(iterator).cuda();q=torch.tensor(qps[offset:offset+4],device='cuda');r=model(x,q,intra,table[q],False,True)
            assert torch.isfinite(r['loss']);(r['loss']/4).backward()
            for k,v in r.items():totals[k]=totals.get(k,0)+float(v.detach())/4
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.2,error_if_nonfinite=True);assert norm>0;optimizer.step()
        update(status='training',updates=step,lr=lr);log('train',step=step,lr=lr,base_qps=qps,gradient_norm=float(norm),**totals)
        if step%100==0:checkpoint(last,step)
        if step%250==0:
            value=validate(screen,'common_surveillance_screen',step);checkpoint(folder/f'candidate_step{step:04d}.pth.tar',step);atomic(folder/f'candidate_step{step:04d}.json',dict(step=step,metrics=value,normalized_score=score(value,base)))
            checkpoint(last,step)
    if not (folder/'candidate_step3000.json').exists():
        value=validate(screen,'common_surveillance_screen',3000);checkpoint(folder/'candidate_step3000.pth.tar',3000);atomic(folder/'candidate_step3000.json',dict(step=3000,metrics=value,normalized_score=score(value,base)));checkpoint(last,3000)
    assert all(p.grad is None for p in intra.parameters())
    torch.save(dict(state_dict=model.state_dict()),folder/'endpoint.pth.tar')
    atomic(folder/'endpoint_audit.json',dict(passed=True,updates=3000,sampler_offset=48000,checkpoint_rule='fixed_update3000_no_performance_fallback',test_used_for_selection=False,endpoint_sha256=hashlib.sha256((folder/'endpoint.pth.tar').read_bytes()).hexdigest()))
    update(status='fixed_endpoint_training_complete',updates=3000,selected_step=3000)
except BaseException:update(status='failed',error=traceback.format_exc());raise
