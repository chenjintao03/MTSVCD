"""Independent true-bitstream validation; never uses the 82 test videos."""
from common import *
import argparse,subprocess,shutil,fcntl,math
p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--gpu',type=int,choices=[4,5],required=True);a=p.parse_args();setup()
BASE=Path('@WORKSPACE@/native420_benchmark');PY='@RT_PYTHON@'
folder=ROOT/a.arm;sel=read(folder/'selection.json')
if not sel['accepted']:
    atomic(folder/'real_validation_gate.json',dict(passed=True,selected_original=True,training_test_leakage=False));raise SystemExit(0)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
shared=ROOT/'validation_stream';shared.mkdir(exist_ok=True)
lock=(shared/'preparation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
manifest=shared/'manifest.json';inputs=shared/'inputs';inputs.mkdir(exist_ok=True)
if not manifest.exists():
    md=read(ROOT/'protocol/self_metadata.json');rows=[]
    for item in read(ROOT/'diagnostic/input_audit.json')['checks']:
        if item['split']!='val':continue
        camera=item['camera'];start=(md[camera]['frames']-96)//2;name=camera+'_validation_middle';target=inputs/(name+'.yuv');src=Path('@DATASET_ROOT@/processed_video')/camera/'clip01.mp4'
        subprocess.run(['/usr/bin/ffmpeg','-v','error','-y','-threads','1','-ss',f'{start/30:.9f}','-i',str(src),'-frames:v','96','-vsync','0','-c:v','rawvideo','-threads','1','-pix_fmt',md[camera]['pix_fmt'],'-f','rawvideo',str(target)],check=True,cwd=ROOT)
        assert target.stat().st_size==96*640*384*3//2
        rows.append(dict(clip=name,camera=camera,frames=96,sha256=sha(target),subset='validation',start=start))
    assert len(rows)==8;atomic(manifest,dict(clips=rows))
fcntl.flock(lock,fcntl.LOCK_UN)
def prepare(tag,checkpoint):
    root=shared/tag
    for n in ['code','protocol','references','inputs','logs']:(root/n).mkdir(exist_ok=True,parents=True)
    shutil.copy2(BASE/'code/common_yuv.py',root/'code/common_yuv.py')
    for n in ['rt_metrics.py','rt_transforms.py']:shutil.copy2(BASE/'references'/n,root/'references'/n)
    for n in ['input_manifest.json','source_sha256.json']:
        shutil.copy2(manifest if n=='input_manifest.json' else BASE/'protocol/source_sha256.json',root/'protocol'/n)
    atomic(root/'protocol/benchmark.json',dict(validation_only=True,qp=[0,21,42,63],ip=32,reset=64,checkpoint_sha256=sha(checkpoint)))
    if not (root/'inputs/yuv420').exists():(root/'inputs/yuv420').symlink_to(inputs,target_is_directory=True)
    source=(BASE/'code/learned_worker.py').read_text()
    needle="w=OLD/'checkpoints'/f'cvpr2025_{name}.pth.tar';assert sha(w)==expected"
    assert source.count(needle)==1
    replacement="w=Path("+repr(str(checkpoint))+") if name=='video' else OLD/'checkpoints'/f'cvpr2025_{name}.pth.tar';assert sha(w)==("+repr(sha(checkpoint))+" if name=='video' else expected)"
    source=source.replace(needle,replacement).replace('if a.pilot:','if True:').replace('independent_decode_checked=a.pilot','independent_decode_checked=True')
    needle='if True:hashes.append(digest(x_hat))'
    assert source.count(needle)==1;source=source.replace(needle,'hashes.append(digest(x_hat))')
    # Add a distortion readout without importing training modules, which disable fused inference.
    needle="  if len(rows)%16==0:update(current=c['clip'],rate=q,frame=len(rows))"
    assert source.count(needle)==1
    extra="""  import torch.nn.functional as F
  from src.utils.transforms import ycbcr2rgb
  src=torch.from_numpy(np.concatenate([y[None],np.stack([u,v]).repeat(2,1).repeat(2,2)],0).copy()).to(device).float()[None]/255
  rec=x_hat.float();dy=(src[:,:1]-rec[:,:1]).square().mean();duv=(F.avg_pool2d(src[:,1:],2,2)-F.avg_pool2d(rec[:,1:],2,2)).square().mean((0,2,3));dyuv=(6*dy+duv.sum())/8;drgb=(ycbcr2rgb(src)-ycbcr2rgb(rec)).square().mean()
  rows[-1]['rd_distortion']=float(.8*dyuv+.2*drgb)
"""
    source=source.replace(needle,extra+needle);(root/'code/learned_worker.py').write_text(source)
    return root
def run(root):
    with (root/'logs/validation.log').open('a') as f:subprocess.run([PY,'-B',str(root/'code/learned_worker.py'),'--method','rt','--gpu',str(a.gpu),'--rates','0','1','2','3'],cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
def metric(root):
    result={}
    for rate,qp in enumerate([0,21,42,63]):
        vals=[]
        for c in read(manifest)['clips']:
            d=read(root/f'results/rt/q{rate}/{c["clip"]}.json');assert d['independent_decode_checked'] and d['input_sha256']==c['sha256']
            assert all(sha(p['path'])==p['sha256'] for p in d['streams'])
            pframes=[f for f in d['per_frame'] if f['type']=='P'];assert len(pframes)==93
            vals.append(sum(f['bytes']*8/(640*384)+math.exp(math.log(768)*qp/63)*[.5,1.2,.5,.9][(f['frame']%32-1)%4]*f['rd_distortion'] for f in pframes)/93)
        result[str(qp)]=sum(vals)/len(vals)
    return result
# A shared immutable baseline is computed once, while the other GPU may wait.
fcntl.flock(lock,fcntl.LOCK_EX)
try:
    baseline_root=prepare('original',P)
    if not (baseline_root/'complete.json').exists():run(baseline_root);atomic(baseline_root/'complete.json',dict(passed=True,metrics=metric(baseline_root)))
    baseline=read(baseline_root/'complete.json')['metrics']
finally:fcntl.flock(lock,fcntl.LOCK_UN)
candidate_root=prepare(a.arm,folder/'selected.pth.tar');run(candidate_root);candidate=metric(candidate_root)
ratios={q:candidate[q]/baseline[q] for q in baseline};score=sum(ratios.values())/4;accepted=score<1 and max(ratios.values())<=1.02
atomic(folder/'estimated_selection.json',sel)
if not accepted:
    shutil.copy2(folder/'selected.pth.tar',folder/'selected_estimated.pth.tar')
    import torch
    torch.save(dict(state_dict=state(P)),folder/'selected.pth.tar');sel.update(accepted=False,step=0,reason='Real validation RD did not pass the predeclared guard; original model selected')
sel.update(real_validation_passed=accepted,real_validation_score=score);atomic(folder/'selection.json',sel)
atomic(folder/'real_validation_gate.json',dict(passed=True,finetuning_accepted=accepted,validation_videos=8,results_per_model=32,frames_per_model=3072,independent_decode_all_points=True,original=baseline,candidate=candidate,ratios=ratios,mean_ratio=score,test_used=False))
