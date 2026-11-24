"""TCM validation uses the same P-frame RD objective as adaptation.

Original-checkpoint caches are shared only when model, input and code signatures
match. All final quality claims still require the independent bitstream guard.
"""
from common import *
import torch,fcntl,math
from adapter import rd_forward
from data import ThreadBatches
def input_signature():
    paths=[ROOT/'protocol'/n for n in ['self_split.json','self_metadata.json','sep_trainlist.txt','sep_testlist.txt']]
    paths += [ROOT/'code'/n for n in ['adapter.py','data.py','validation.py']]
    paths += [ROOT/'references/rt_transforms.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}
@torch.no_grad()
def validate(model,intra,quality,dataset,progress):
    model.eval();sums=dict(loss=0.,p_bpp_estimated=0.,p_rgb_mse=0.,p_psnr_rgb8=0.);windows=0;pframes=0
    for frames in ThreadBatches(dataset,1,2,2):
        frames=frames.cuda();ref=feature=None;values={k:[] for k in sums}
        for n in range(frames.shape[1]):
            x=frames[:,n]
            if n%32==0:ref=intra(x)['x_hat'].clamp(0,1);feature=None;continue
            out=rd_forward(model,x,ref,feature,False,False);ref=out['recon_image'].clamp(0,1);feature=out['feature']
            values['loss'].append(float(out['bpp']+LAMBDAS[quality]*out['mse_loss']));values['p_bpp_estimated'].append(float(out['bpp']));values['p_rgb_mse'].append(float(out['mse_loss']))
            rounded=(ref*255).round().clamp(0,255)/255
            values['p_psnr_rgb8'].append(float(-10*torch.log10((rounded-x).square().mean().clamp_min(1e-16))))
        assert len(values['loss'])==(93 if frames.shape[1]==96 else 4)
        for k in sums:sums[k]+=sum(values[k])/len(values[k])
        windows+=1;pframes+=len(values['loss'])
        if windows%20==0:progress(windows,len(dataset))
    assert windows==len(dataset)
    result={k:v/windows for k,v in sums.items()};assert all(math.isfinite(v) for v in result.values())
    result.update(windows=windows,p_frames=pframes,rate_kind='estimated probability rate, not actual bitstream',i_cost_excluded=True)
    model.train();return result
def cached_validation(model,intra,quality,dataset,progress,path,checkpoint,shared_original=False):
    signature=dict(inputs=input_signature(),quality=quality,domain=dataset.domain,videos=len(dataset.ids),windows=len(dataset),checkpoint_sha256=sha(checkpoint),i_sha256=sha(WEIGHTS/f'intra_psnr_TMM_q{quality}.pth.tar'))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    lock=path.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        if path.exists():
            d=read(path);assert d['signature']==signature,'Do not reuse validation across different models or inputs';return d['metrics']
        result=validate(model,intra,quality,dataset,progress);atomic(path,dict(signature=signature,metrics=result,shared_original=shared_original));return result
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
