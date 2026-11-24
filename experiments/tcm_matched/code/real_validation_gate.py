"""Per-quality true-bitstream checkpoint protection on eight validation scenes.

Generated workers copy the audited native420 benchmark adapter; each point is
decoded again with fresh I/P instances before its RGB-domain P-only RD is used.
No test frames are read here.
"""
from common import *
import argparse,subprocess,shutil,fcntl,math
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['surveillance','vimeo'],required=True);p.add_argument('--quality',type=int,choices=[1,2,3,4],required=True);p.add_argument('--gpu',type=int,choices=[4,5],required=True);a=p.parse_args();setup()
folder=ROOT/a.arm/f'q{a.quality}';selection=read(folder/'estimated_selection.json');shared=ROOT/'validation_stream';shared.mkdir(exist_ok=True)
lock=(shared/f'q{a.quality}.lock').open('a')

def prepare_inputs():
    import random
    guard=(shared/'input.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX)
    try:
        manifest=shared/'manifest.json'
        if manifest.exists():
            value=read(manifest);assert len(value['clips'])==8 and all(sha(shared/'inputs'/f'{c["clip"]}.yuv')==c['sha256'] for c in value['clips']);return
        rows=[r for r in read(ROOT/'protocol/self_split.json')['records'] if r['subset']=='short' and r['split']=='val'];random.Random(20261105).shuffle(rows);rows=rows[:8]
        metadata=read(ROOT/'protocol/self_metadata.json');inputs=shared/'inputs';inputs.mkdir(exist_ok=True);clips=[]
        for row in rows:
            camera=row['camera'];start=(metadata[camera]['frames']-96)//2;name=camera+'_validation_middle';target=inputs/f'{name}.yuv';src=Path('@DATASET_ROOT@/processed_video')/camera/'clip01.mp4'
            subprocess.run(['/usr/bin/ffmpeg','-v','error','-y','-threads','1','-ss',f'{start/30:.9f}','-i',str(src),'-frames:v','96','-vsync','0','-c:v','rawvideo','-threads','1','-pix_fmt',metadata[camera]['pix_fmt'],'-f','rawvideo',str(target)],cwd=ROOT,check=True)
            assert target.stat().st_size==96*640*384*3//2
            clips.append(dict(clip=name,camera=camera,start=start,frames=96,subset='validation',sha256=sha(target)))
        atomic(manifest,dict(clips=clips,selection='seed20261105 eight surveillance val scenes, middle96 only',test_used=False))
    finally:fcntl.flock(guard,fcntl.LOCK_UN);guard.close()

def prepare(tag,checkpoint):
    root=shared/tag/f'q{a.quality}'
    for n in ['code','protocol','references','inputs','logs']:(root/n).mkdir(parents=True,exist_ok=True)
    shutil.copy2(BASE/'code/common_yuv.py',root/'code/common_yuv.py')
    for n in ['rt_metrics.py','rt_transforms.py']:shutil.copy2(BASE/'references'/n,root/'references'/n)
    shutil.copy2(shared/'manifest.json',root/'protocol/input_manifest.json');shutil.copy2(BASE/'protocol/source_sha256.json',root/'protocol/source_sha256.json')
    protocol=dict(validation_only=True,quality=a.quality,lambda_value=LAMBDAS[a.quality],ip=32,frames=96,p_checkpoint_sha256=sha(checkpoint),i_sha256=sha(WEIGHTS/f'intra_psnr_TMM_q{a.quality}.pth.tar'),worker_base_sha256=sha(BASE/'code/learned_worker.py'))
    path=root/'protocol/benchmark.json'
    if path.exists():assert read(path)==protocol
    else:atomic(path,protocol)
    if not (root/'inputs/yuv420').exists():(root/'inputs/yuv420').symlink_to(shared/'inputs',target_is_directory=True)
    source=(BASE/'code/learned_worker.py').read_text()
    needle="iw=W/f'intra_psnr_TMM_q{q+1}.pth.tar';pw=W/f'inter_psnr_TMM_q{q+1}.pth'"
    assert source.count(needle)==1
    source=source.replace(needle,"iw=W/f'intra_psnr_TMM_q{q+1}.pth.tar';pw=Path("+repr(str(checkpoint))+");assert q=="+str(a.quality-1)+" and sha(pw)=="+repr(sha(checkpoint)))
    source=source.replace('if a.pilot:','if True:').replace('independent_decode_checked=a.pilot','independent_decode_checked=True')
    needle="   row=dict(frame=n,type='I' if n%32==0 else 'P',bytes=bit.stat().st_size,**distortion((y,u,v),(yr,uvr[0],uvr[1])))"
    assert source.count(needle)==1
    source=source.replace(needle,needle+"\n   row['p_rgb_mse']=float((ref-x).square().mean())")
    compile(source,str(root/'code/learned_worker.py'),'exec');target=root/'code/learned_worker.py'
    if target.exists():assert target.read_text()==source
    else:target.write_text(source)
    return root

def metric(root):
    values=[];evidence=[]
    for c in read(shared/'manifest.json')['clips']:
        d=read(root/f'results/tcm/q{a.quality-1}/{c["clip"]}.json');assert d['independent_decode_checked'] and d['input_sha256']==c['sha256'] and d['protocol_sha256']==sha(root/'protocol/benchmark.json')
        assert d['frames']==96 and all(sha(s['path'])==s['sha256'] for s in d['streams'])
        rows=[r for r in d['per_frame'] if r['type']=='P'];assert len(rows)==93
        value=sum(r['bytes']*8/(640*384)+LAMBDAS[a.quality]*r['p_rgb_mse'] for r in rows)/93;assert math.isfinite(value);values.append(value);evidence.append(dict(clip=c['clip'],loss=value,result_sha256=sha(root/f'results/tcm/q{a.quality-1}/{c["clip"]}.json')))
    return dict(loss=sum(values)/8,windows=8,frames=768,p_frames=744,independent_decode_all_points=True,objective='actual P file bytes/pixels + lambda*canonical RGB MSE; I excluded',evidence=evidence)

def run(tag,checkpoint):
    root=prepare(tag,checkpoint)
    with (root/'logs/validation.log').open('a') as f:subprocess.run([PY,'-B',str(root/'code/learned_worker.py'),'--method','tcm','--gpu',str(a.gpu),'--rates',str(a.quality-1)],cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
    result=metric(root);atomic(root/'complete_audit.json',dict(passed=True,metrics=result));return result

assert read(RT_ROOT/'complete_audit.json')['passed']
assert all(read(ROOT/f'diagnostic/q{q}/gradient_gate.json')['status']=='passed' for q in [1,2,3,4])
original=WEIGHTS/f'inter_psnr_TMM_q{a.quality}.pth';accepted=False;report=dict(passed=True,test_used=False,quality=a.quality,selected_original=True)
if selection['accepted']:
    prepare_inputs();fcntl.flock(lock,fcntl.LOCK_EX)
    try:baseline=run('original',original)
    finally:fcntl.flock(lock,fcntl.LOCK_UN)
    candidate=run(a.arm,folder/'selected_estimated.pth.tar');ratio=candidate['loss']/baseline['loss'];accepted=ratio<1
    report.update(original=baseline,candidate=candidate,ratio=ratio,finetuning_accepted=accepted,selected_original=not accepted)
import torch
chosen=folder/'selected_estimated.pth.tar' if accepted else original
torch.save(dict(state_dict=state(chosen)),folder/'selected.pth.tar')
selection.update(accepted=accepted,step=selection['step'] if accepted else 0,real_validation_passed=accepted,selected_original=not accepted)
atomic(folder/'selection.json',selection);atomic(folder/'real_validation_gate.json',report)
