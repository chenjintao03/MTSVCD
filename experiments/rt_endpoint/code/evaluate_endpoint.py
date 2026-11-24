from common import *
import argparse,subprocess,shutil,collections,traceback
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['vimeo','shanghai','ua_detrac'],required=True);p.add_argument('--gpu',type=int,choices=[4,5],required=True);a=p.parse_args();setup(a.gpu)
BASE=Path('@WORKSPACE@/native420_benchmark');target=ROOT/'evaluation'/a.arm;selected=ROOT/a.arm/'endpoint.pth.tar'
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
try:
    assert read(ROOT/'protocol/experiment.json')['checkpoint_rule']=='fixed_update3000_no_performance_fallback'
    target.mkdir(parents=True,exist_ok=True)
    for d in ['code','protocol','inputs','references']:(target/d).mkdir(exist_ok=True)
    shutil.copy2(BASE/'code/common_yuv.py',target/'code/common_yuv.py')
    source=(BASE/'code/learned_worker.py').read_text();needle="w=OLD/'checkpoints'/f'cvpr2025_{name}.pth.tar';assert sha(w)==expected"
    replacement="w=Path("+repr(str(selected))+") if name=='video' else OLD/'checkpoints'/f'cvpr2025_{name}.pth.tar';assert sha(w)==("+repr(digest(selected))+" if name=='video' else expected)"
    assert source.count(needle)==1;source=source.replace(needle,replacement)
    (target/'code/learned_worker.py').write_text(source)
    for name in ['input_manifest.json','source_sha256.json']:shutil.copy2(BASE/'protocol'/name,target/'protocol'/name)
    cfg=read(BASE/'protocol/benchmark.json');cfg.update(finetune_checkpoint_sha256=digest(selected),training_arm=a.arm,checkpoint_rule='fixed_update3000_no_performance_fallback');atomic(target/'protocol/benchmark.json',cfg)
    for name in ['rt_metrics.py','rt_transforms.py']:shutil.copy2(BASE/'references'/name,target/'references'/name)
    link=target/'inputs/yuv420'
    if not link.exists():link.symlink_to(BASE/'inputs/yuv420',target_is_directory=True)
    cmd=['@RT_PYTHON@','-B',str(target/'code/learned_worker.py'),'--method','rt','--gpu',str(a.gpu),'--rates','0','1','2','3']
    with (target/'test.log').open('a') as f:
        subprocess.run(cmd+['--pilot'],cwd=target,stdout=f,stderr=subprocess.STDOUT,check=True)
        for q in range(4):
            r=read(target/f'results/rt/q{q}/cam0001_begin.json');assert r['independent_decode_checked'] and len(r['per_frame'])==96
        atomic(target/'endpoint_pilot_audit.json',dict(passed=True,results=4,frames=384,fresh_independent_decode=True,endpoint_sha256=digest(selected)))
        subprocess.run(cmd,cwd=target,stdout=f,stderr=subprocess.STDOUT,check=True)
    curves=[];total=0
    for q in range(4):
        by=collections.defaultdict(list)
        for c in read(BASE/'protocol/input_manifest.json')['clips']:
            r=read(target/f'results/rt/q{q}/{c["clip"]}.json');assert r['input_sha256']==c['sha256'] and r['true_bitstream'] and len(r['per_frame'])==96 and r['i_positions']==[0,32,64]
            for s in r['streams']:assert digest(s['path'])==s['sha256']
            by[r['camera']].append(r);total+=1
        assert len(by)==82 and all(len(rs)==3 for rs in by.values())
        curves.append(dict(quality=q,videos=82,**{k:sum(sum(r[k] for r in rs)/3 for rs in by.values())/82 for k in ['bpp','psnr','psnr_y','psnr_u','psnr_v','ms_ssim']}))
    atomic(target/'rd_summary.json',dict(arm=a.arm,curves=curves,updates=3000,checkpoint_rule='fixed_update3000_no_performance_fallback',test_used_for_selection=False))
    atomic(target/'complete_audit.json',dict(passed=True,results=total,frames=total*96,all_stream_sha256_verified=True,test_used_for_selection=False,worse_than_original_retained=True,checkpoint_sha256=digest(selected)))
except BaseException:atomic(target/'failure.json',dict(error=traceback.format_exc(),fixed_checkpoint_retained=True,no_fallback=True));raise
