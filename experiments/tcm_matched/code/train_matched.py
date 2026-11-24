"""Equal-budget per-quality TCM adaptation, pending the four GPU gates."""
from common import *
import argparse,random,fcntl,traceback
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[4,5],required=True);p.add_argument('--arm',choices=['surveillance','vimeo'],required=True);p.add_argument('--quality',type=int,choices=[1,2,3,4],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();setup(a.gpu)
import torch,numpy as np
from adapter import models,sequence_loss
from data import Samples,Validation,ThreadBatches
from validation import cached_validation
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
seed=20261105+a.quality;torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
folder=ROOT/a.arm/f'q{a.quality}';folder.mkdir(parents=True,exist_ok=True);lock=(folder/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
status=dict(status='starting',pid=os.getpid(),arm=a.arm,quality=a.quality,gpu=a.gpu,updates=0,target_updates=3000)
cfg=read(ROOT/'protocol/experiment.json')
def update(**kw):status.update(kw,updated=time.strftime('%Y-%m-%d %H:%M:%S'));atomic(folder/'status.json',status)
def log(event,**kw):
    with (folder/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(event=event,time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw))+'\n')
def sources():return {n:sha(ROOT/'code'/n) for n in ['common.py','adapter.py','data.py','validation.py','train_matched.py']}
def save(path,step):
    tmp=path.with_suffix('.tmp');torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),updates=step,protocol=cfg,source_sha256=sources(),quality=a.quality,arm=a.arm,sampler_offset=step*4,python_random=random.getstate(),numpy_random=np.random.get_state(),torch_random=torch.get_rng_state(),cuda_random=torch.cuda.get_rng_state_all(),frozen_i_sha256=meta['i_sha256']),tmp);tmp.replace(path)
def validate(ds,label,checkpoint,path,shared=False,checkpoint_step=None):
    update(status='validating',validation=label,validation_checkpoint_step=checkpoint_step,validation_windows=len(ds))
    def progress(done,total):update(validation_done=done,validation_windows=total)
    value=cached_validation(model,intra,a.quality,ds,progress,path,checkpoint,shared);log('validation',checkpoint_step=checkpoint_step,label=label,metrics=value);return value
try:
    assert cfg['training_ready'] and cfg['frames']==5 and cfg['effective_batch']==4 and cfg['updates_per_quality_per_arm']==3000
    assert read(RT_ROOT/'complete_audit.json')['passed']
    assert all(read(ROOT/f'diagnostic/q{q}/gradient_gate.json')['status']=='passed' for q in [1,2,3,4])
    assert torch.cuda.device_count()==1
    if hasattr(torch.cuda.get_device_properties(0),'uuid'):assert str(torch.cuda.get_device_properties(0).uuid).lower().replace('gpu-','')==UUID[a.gpu].lower().replace('gpu-','')
    intra,model,meta=models(a.quality);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
    original=WEIGHTS/f'inter_psnr_TMM_q{a.quality}.pth';last=folder/'last.pth.tar'
    assert a.resume or not last.exists(),'Explicit audited resume required; never overwrite a run'
    screen=Validation('surveillance',limit=8);full=Validation('surveillance')
    # The model is still original here, including on resume; cached baselines are shared across arms.
    base=validate(screen,'common_original_screen',original,ROOT/f'original_validation/q{a.quality}/screen.json',True,0)
    start=0
    if last.exists():
        d=torch.load(last,map_location='cpu',weights_only=False)
        assert d['protocol']==cfg and d['source_sha256']==sources() and d['quality']==a.quality and d['arm']==a.arm and d['frozen_i_sha256']==meta['i_sha256'] and d['sampler_offset']==d['updates']*4
        model.load_dict(d['state_dict'],strict=True);optimizer.load_state_dict(d['optimizer']);start=d['updates'];random.setstate(d['python_random']);np.random.set_state(d['numpy_random']);torch.set_rng_state(d['torch_random']);torch.cuda.set_rng_state_all(d['cuda_random']);log('resume',updates=start,sampler_offset=start*4)
    update(updates=start);loader=iter(ThreadBatches(Samples(a.arm,a.quality,3000,start*4),1,2,4))
    for step in range(start+1,3001):
        lr=1e-5 if step<=2000 else 5e-6
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True);totals={}
        for _ in range(4):
            frames=next(loader).cuda();out=sequence_loss(model,intra,frames,a.quality,True,True);assert torch.isfinite(out['loss']);(out['loss']/4).backward()
            for k,v in out.items():totals[k]=totals.get(k,0)+float(v.detach())/4
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);assert norm>0 and all(p.grad is None for p in intra.parameters());optimizer.step()
        update(status='training',updates=step,lr=lr);log('train',step=step,lr=lr,gradient_norm=float(norm),**totals)
        if step%100==0:save(last,step)
        if step%250==0:
            path=folder/f'candidate_step{step:04d}.pth.tar';save(path,step)
            value=validate(screen,'common_candidate_screen',path,folder/f'screen_step{step:04d}.json',False,step)
            atomic(folder/f'candidate_step{step:04d}.json',dict(step=step,metrics=value,normalized_score=value['loss']/base['loss']));save(last,step)
    update(updates=3000)
    # Complete the final screen if a prior interruption followed its update checkpoint.
    if not (folder/'candidate_step3000.json').exists():
        path=folder/'candidate_step3000.pth.tar';save(path,3000);value=validate(screen,'common_candidate_screen',path,folder/'screen_step3000.json',False,3000);atomic(folder/'candidate_step3000.json',dict(step=3000,metrics=value,normalized_score=value['loss']/base['loss']));save(last,3000)
    model.load_dict(state(original),strict=True)
    full_base=validate(full,'common_original_full',original,ROOT/f'original_validation/q{a.quality}/full.json',True,0)
    shortlist=sorted([read(p) for p in folder.glob('candidate_step*.json')],key=lambda d:(d['normalized_score'],d['step']))[:2]
    best=None;best_score=1.;results=[]
    for item in shortlist:
        step=item['step'];path=folder/f'candidate_step{step:04d}.pth.tar';model.load_dict(state(path),strict=True)
        value=validate(full,'common_candidate_full',path,folder/f'full_step{step:04d}.json',False,step);score=value['loss']/full_base['loss'];okay=score<1
        results.append(dict(step=step,metrics=value,normalized_score=score,eligible=okay))
        if okay and score<best_score:best=step;best_score=score
    model.load_dict(state(folder/f'candidate_step{best:04d}.pth.tar') if best else state(original),strict=True)
    torch.save(dict(state_dict=model.state_dict()),folder/'selected_estimated.pth.tar')
    atomic(folder/'estimated_selection.json',dict(accepted=best is not None,step=best or 0,updates_attempted=3000,quality=a.quality,original=full_base,candidates=results,common_validation_videos=81,common_validation_windows=243,test_used_for_selection=False,estimated_rate_only=True,real_guard_required=True))
    assert sha(WEIGHTS/f'intra_psnr_TMM_q{a.quality}.pth.tar')==meta['i_sha256'] and sha(original)==meta['p_original_sha256']
    update(status='estimated_selection_complete',updates=3000,selected_estimated_step=best or 0,real_guard_pending=True)
except BaseException:update(status='failed',error=traceback.format_exc());raise
