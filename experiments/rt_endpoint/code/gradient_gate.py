from common import *
import argparse,math,traceback
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[4,5],required=True);p.add_argument('--arm',choices=['shanghai','ua_detrac'],required=True);a=p.parse_args();setup(a.gpu)
import torch
from model import Video,IntraForTraining
from src.models.image_model import DMCI
from src.models.video_model import DMC
from data import Clips
torch.set_num_threads(2);torch.manual_seed(20261104)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
report=dict(status='running',gpu=a.gpu,checks=[])
def save():atomic(ROOT/f'diagnostic/{a.arm}_gradient_gate.json',report)
def load(cls,path):
    m=cls().cuda();m.load_state_dict(state(path),strict=True);return m
save()
try:
    assert read(ROOT/'diagnostic/input_audit.json')['status']=='passed'
    assert str(torch.cuda.get_device_properties(0).uuid).lower().replace('gpu-','')==os.environ['CUDA_VISIBLE_DEVICES'].lower().replace('gpu-','')
    i=load(IntraForTraining,I).eval();i.requires_grad_(False);m=load(Video,P)
    for arm in [a.arm]:
        ds=Clips(arm,'train',steps=1,seed=20261104);frames=ds[0][None].cuda()
        errors=[];rates=[]
        with torch.no_grad():
            for qp in [0,21,42,63]:
                q=torch.tensor([qp],device='cuda');oi=load(DMCI,I).eval();ov=load(DMC,P).eval();dec=load(DMC,P).eval()
                oi.update(.12);ov.update(.12);dec.update(.12);sps=dict(height=256,width=256,ec_part=0,use_ada_i=0)
                ti=i(frames[:,0],q);e=oi.compress(frames[:,0],qp);di=oi.decompress(e['bit_stream'],sps,qp)['x_hat'];assert (ti-di).abs().max()<1e-5
                m.clear_dpb();m.add_ref_frame(frame=ti);ov.clear_dpb();ov.add_ref_frame(frame=di);dec.clear_dpb();dec.add_ref_frame(frame=di)
                estimates=[];actual=[]
                for n in range(1,7):
                    qq=qp+m.qp_shift[m.index_map[n%8]];pred=m.forward_one_frame(frames[:,n],torch.tensor([qq],device='cuda'),False)
                    enc=ov.compress(frames[:,n],qq);out=dec.decompress(enc['bit_stream'],sps,qq)['x_hat'];error=float((pred['x_hat']-out).abs().max());assert error<2e-4
                    errors.append(error);estimates.append(float(pred['bpp'].mean()));actual.append(len(enc['bit_stream'])*8/(256*256))
                rates.append(dict(qp=qp,estimated_p_bpp=sum(estimates)/6,actual_payload_p_bpp=sum(actual)/6))
                del oi,ov,dec
        assert all(rates[n]['estimated_p_bpp']<rates[n+1]['estimated_p_bpp'] and rates[n]['actual_payload_p_bpp']<rates[n+1]['actual_payload_p_bpp'] for n in range(3))
        # Last-P loss must reach the first P input through temporal references.
        m.zero_grad(set_to_none=True);x=frames.detach().requires_grad_(True);captured=[];fn=m.forward_one_frame
        def record(*args,**kwargs):
            value=fn(*args,**kwargs);captured.append(value);return value
        m.forward_one_frame=record
        q=torch.tensor([21],device='cuda');lam=torch.tensor([math.exp(math.log(768)*21/63)],device='cuda')
        r=m(x,q,i,lam,detach_refs=False,noisy_rate=False)
        (captured[-1]['bpp']+lam*captured[-1]['distortion']).mean().backward()
        temporal=float(x.grad[:,1].abs().sum());assert temporal>0 and torch.isfinite(x.grad).all()
        m.forward_one_frame=fn;m.zero_grad(set_to_none=True);del x,r,captured
        # A full seven-frame clip contributes to each update; I stays frozen.
        batch=torch.stack([ds[n] for n in range(4)]).cuda();q=torch.tensor([0,21,42,63],device='cuda');lam=torch.exp(q*math.log(768)/63)
        loss=m(batch,q,i,lam,False,False)['loss'];loss.backward();direct={k:v.grad.detach().clone() for k,v in m.named_parameters() if v.grad is not None}
        m.zero_grad(set_to_none=True)
        for n in range(4):(m(batch[n:n+1],q[n:n+1],i,lam[n:n+1],False,False)['loss']/4).backward()
        squared=sum(float((p.grad-direct[k]).double().square().sum()) for k,p in m.named_parameters() if k in direct)
        norm=sum(float(v.double().square().sum()) for v in direct.values())**.5
        relative=squared**.5/max(norm,1e-12);assert relative<.002,(arm,relative)
        assert all(p.grad is None for p in i.parameters())
        gn=torch.nn.utils.clip_grad_norm_(m.parameters(),.2,error_if_nonfinite=True);assert gn>0
        optimizer=torch.optim.Adam(m.parameters(),lr=1e-6);optimizer.step()
        path=ROOT/f'diagnostic/{arm}_one_update.pth.tar';path.parent.mkdir(exist_ok=True);torch.save(dict(state_dict=m.state_dict()),path)
        checked=load(DMC,path);checked.update(.12)
        # Check the changed P model's reconstruction against an independent real decoder.
        with torch.no_grad():
            qq=21;q1=torch.tensor([qq],device='cuda');first=i(frames[:,0],q1);m.clear_dpb();m.add_ref_frame(frame=first)
            encnet=load(DMC,path);encnet.update(.12);encnet.add_ref_frame(frame=first);checked.clear_dpb();checked.add_ref_frame(frame=first)
            for n in range(1,7):
                qp=qq+m.qp_shift[m.index_map[n%8]];pred=m.forward_one_frame(frames[:,n],torch.tensor([qp],device='cuda'),False)['x_hat'];stream=encnet.compress(frames[:,n],qp)['bit_stream'];rec=checked.decompress(stream,sps,qp)['x_hat'];assert (pred-rec).abs().max()<2e-4
        report['checks'].append(dict(arm=arm,baseline_all_four_qp_matches_real_decode=True,max_abs_reconstruction_error=max(errors),last_p_loss_reaches_first_p_input=temporal,gradient_accumulation_relative_l2_error=relative,frozen_i_no_gradient=True,one_update_official_load_and_independent_decode=True,estimated_and_actual_rate=rates,peak_memory_bytes=torch.cuda.max_memory_allocated()))
        save();del checked,encnet,optimizer,batch,loss,direct,frames,ds
        m=load(Video,P);torch.cuda.empty_cache()
    report.update(status='passed');save()
except BaseException:report.update(status='failed',error=traceback.format_exc());save();raise
