"""Pending GPU gate; launch only after the matched RT work releases GPUs4/5."""
from common import *
import argparse,traceback,math
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[4,5],required=True);p.add_argument('--quality',type=int,choices=[1,2,3,4],required=True);a=p.parse_args();setup(a.gpu)
assert read(RT_ROOT/'complete_audit.json')['passed'],'Do not overlap active RT work'
import torch
import adapter
from adapter import models,rd_forward,sequence_loss,DMC,IntraNoAR
from data import Samples,self_ids,self_frames,colour,META
from src.utils.stream_helper import decoder_p,decode_i,get_downsampled_shape
torch.set_num_threads(2);torch.manual_seed(20261105+a.quality)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
OUT=ROOT/f'diagnostic/q{a.quality}';OUT.mkdir(parents=True,exist_ok=True)
report=dict(status='running',gpu=a.gpu,gpu_uuid=UUID[a.gpu],quality=a.quality,frames=5,checks=[],source_signature={n:sha(ROOT/'code'/n) for n in ['common.py','data.py','adapter.py','gradient_gate.py']})
def save():atomic(OUT/'gradient_gate.json',report)
def state_hash(m):
    h=hashlib.sha256()
    for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
@torch.no_grad()
def streams(m,i,frames,label,checkpoint=None):
    m.eval();m.update(force=True);i.update(force=True)
    di,dp,_=models(a.quality,checkpoint);di.update(force=True);dp.eval();dp.update(force=True)
    if checkpoint is None:dp.load_dict(m.state_dict(),strict=True);dp.update(force=True)
    ref=feature=dref=dfeature=None;rows=[]
    for n in range(5):
        x=frames[:,n];path=OUT/f'{label}_{n}.bin'
        if n==0:
            r=i.encode_decode(x,str(path),pic_height=256,pic_width=256);ref=r['x_hat'].clamp(0,1)
            h,w,y,z=decode_i(str(path));dref=di.decompress([[y],[z]],get_downsampled_shape(h,w,64))['x_hat'].clamp(0,1)
            assert (ref-dref).abs().max()<=2e-6
        else:
            f=m.forward_one_frame(x,ref,feature);adapt=rd_forward(m,x,ref,feature,False,False)
            assert (f['recon_image']-adapt['recon_image']).abs().max()<=2e-6
            assert (f['feature']-adapt['feature']).abs().max()<=2e-6
            r=m.encode_decode(x,ref,feature,str(path),pic_height=256,pic_width=256);d=dp.decompress(dref,dfeature,decoder_p(str(path)),256,256)
            error=float((r['x_hat']-d['x_hat']).abs().max());fe=float((r['feature']-d['feature']).abs().max())
            assert error<=2e-6 and fe<=2e-6 and (f['recon_image'].clamp(0,1)-r['x_hat']).abs().max()<=2e-6
            ref=r['x_hat'].clamp(0,1);feature=r['feature'];dref=d['x_hat'].clamp(0,1);dfeature=d['feature']
            rows.append(dict(frame=n,bytes=path.stat().st_size,independent_decode_max_error=error,feature_error=fe))
        assert torch.isfinite(ref).all()
    del di,dp;torch.cuda.empty_cache();m.train();return rows
save()
try:
    assert read(ROOT/'diagnostic/input_gate.json')['status']=='passed' and read(ROOT/'diagnostic/cpu_model_gate.json')['status']=='passed'
    assert torch.cuda.device_count()==1
    prop=torch.cuda.get_device_properties(0)
    if hasattr(prop,'uuid'):assert str(prop.uuid).lower().replace('gpu-','')==UUID[a.gpu].lower().replace('gpu-','')
    # Verify the CPU training input against the actual GPU formal adapter, not only its CPU formula.
    r=self_ids('val')[0];src=Path('@DATASET_ROOT@/processed_video')/r['camera']/'clip01.mp4'
    import subprocess,numpy as np
    b=subprocess.check_output(['/usr/bin/ffmpeg','-v','error','-threads','1','-i',str(src),'-frames:v','5','-c:v','rawvideo','-threads','1','-pix_fmt',META[r['camera']]['pix_fmt'],'-f','rawvideo','-'])
    raw=np.frombuffer(b,np.uint8).reshape(5,368640);y=raw[:,:245760].reshape(5,1,384,640);uv=np.stack([raw[:,245760:307200].reshape(5,192,320),raw[:,307200:].reshape(5,192,320)],1)
    x=torch.cat([torch.from_numpy(y.copy()),torch.from_numpy(uv.copy()).repeat_interleave(2,2).repeat_interleave(2,3)],1).cuda().float()/255
    gpu=torch.round(colour.ycbcr2rgb(x)*255).clamp(0,255)/255
    canonical_device=os.environ['TCM_CANONICAL_RGB8_DEVICE'];os.environ['TCM_CANONICAL_RGB8_DEVICE']='cpu'
    try:old_cpu=self_frames(r,0,5)
    finally:os.environ['TCM_CANONICAL_RGB8_DEVICE']=canonical_device
    diff=(old_cpu.cuda()-gpu).abs();report['old_cpu_formula_discrepant_values']=int(torch.count_nonzero(diff));report['old_cpu_formula_max_rgb8_code_error']=float(diff.max()*255)
    report['old_cpu_actual_integer_rgb8_code_differences']=int(torch.count_nonzero(torch.round(old_cpu.cuda()*255)-torch.round(gpu*255)))
    cpu=self_frames(r,0,5).cuda();assert torch.equal(cpu,gpu),'Staged canonical training inputs differ from official GPU RGB8'
    report['canonical_rgb8_staged_input_gpu_equal']=True;report['canonical_conversion']='official GPU BT709/RGB8, CPU staging before shared crop';del x,gpu,cpu,old_cpu,diff
    for arm in ['surveillance','vimeo']:
        i,m,meta=models(a.quality);i.update(force=True);before_i=state_hash(i);ds=Samples(arm,a.quality);frames=ds[0][None].cuda()
        original_streams=streams(m,i,frames,arm+'_original')
        # The last P loss must reach the first P input, through propagated reference state.
        x=frames.detach().requires_grad_(True);captured=[];base=adapter.rd_forward
        def capture(*args,**kwargs):
            out=base(*args,**kwargs);captured.append(out);return out
        adapter.rd_forward=capture
        try:
            out=sequence_loss(m,i,x,a.quality,True,False);last=captured[-1];(last['bpp']+LAMBDAS[a.quality]*last['mse_loss']).backward()
            reach=float(x.grad[:,1].abs().sum());assert reach>0 and torch.isfinite(x.grad).all()
        finally:adapter.rd_forward=base
        m.zero_grad(set_to_none=True);del x,captured,out,last
        # Diagnose batch-shape arithmetic separately from actual micro1 accumulation.
        assert not any(isinstance(n,(torch.nn.BatchNorm1d,torch.nn.BatchNorm2d,torch.nn.BatchNorm3d,torch.nn.SyncBatchNorm)) for n in m.modules())
        batch=torch.stack([ds[0],ds[1]]).cuda();loss=sequence_loss(m,i,batch,a.quality,True,False)['loss'];loss.backward();direct={k:v.grad.detach().clone() for k,v in m.named_parameters() if v.grad is not None};m.zero_grad(set_to_none=True)
        for n in range(2):(sequence_loss(m,i,batch[n:n+1],a.quality,True,False)['loss']/2).backward()
        denom=sum(float(v.double().square().sum()) for v in direct.values())**.5
        shape_error=sum(float((v.grad-direct[k]).double().square().sum()) for k,v in m.named_parameters() if k in direct)**.5/max(denom,1e-12);assert math.isfinite(shape_error)
        del direct,loss;m.zero_grad(set_to_none=True)
        # The implemented effective4 objective is the mean of four micro1 graphs.
        # Compare one backward through their sum with four accumulated backwards,
        # keeping the network batch shape identical on both sides of the check.
        batch=torch.stack([ds[n] for n in range(4)]).cuda()
        loss=sum(sequence_loss(m,i,batch[n:n+1],a.quality,True,False)['loss']/4 for n in range(4));loss.backward();direct={k:v.grad.detach().clone() for k,v in m.named_parameters() if v.grad is not None};m.zero_grad(set_to_none=True)
        for n in range(4):(sequence_loss(m,i,batch[n:n+1],a.quality,True,False)['loss']/4).backward()
        denom=sum(float(v.double().square().sum()) for v in direct.values())**.5
        error=sum(float((v.grad-direct[k]).double().square().sum()) for k,v in m.named_parameters() if k in direct)**.5/max(denom,1e-12);assert error<2e-5
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);assert norm>0 and all(p.grad is None for p in i.parameters())
        opt=torch.optim.AdamW(m.parameters(),lr=1e-6,weight_decay=1e-4);opt.step();m.update(force=True);path=OUT/f'{arm}_one_update.pth.tar';torch.save(dict(state_dict=m.state_dict()),path)
        updated_streams=streams(m,i,frames,arm+'_updated',path);assert state_hash(i)==before_i
        report['checks'].append(dict(arm=arm,last_p_to_first_p_input_gradient=reach,physical_batch2_vs_two_micro1_gradient_relative_l2_diagnostic=shape_error,four_micro1_joint_vs_accumulated_gradient_relative_l2=error,accumulation_gate_threshold=2e-5,effective_batch_planned=4,batch_dependent_layers_absent=True,no_claim_of_bit_exact_physical_batch4_equivalence=True,frozen_i_state_unchanged=True,original=original_streams,updated=updated_streams,export_official_load=True,original_weights=meta,peak_memory_bytes=torch.cuda.max_memory_allocated()));save()
        del i,m,frames,ds,opt,batch,loss,direct;torch.cuda.empty_cache()
    report.update(status='passed');save()
except BaseException:report.update(status='failed',error=traceback.format_exc());save();raise
