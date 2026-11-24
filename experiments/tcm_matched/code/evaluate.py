"""Frozen82x3x96 native420 test; four paired I models and validation-selected P models."""
from common import *
import argparse,shutil,subprocess,collections
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['surveillance','vimeo'],required=True);p.add_argument('--gpu',type=int,choices=[4,5],required=True);a=p.parse_args();setup(a.gpu)
target=ROOT/'evaluation'/a.arm;target.mkdir(parents=True,exist_ok=True);selected={};selections={}
for q in [1,2,3,4]:
    folder=ROOT/a.arm/f'q{q}';sel=read(folder/'selection.json');assert read(folder/'real_validation_gate.json')['passed'] and read(folder/'natural_status.json')['status']=='complete'
    selections[q]=sel;selected[q]=folder/'selected.pth.tar'
for d in ['code','protocol','references','inputs']:(target/d).mkdir(exist_ok=True)
shutil.copy2(BASE/'code/common_yuv.py',target/'code/common_yuv.py')
source=(BASE/'code/learned_worker.py').read_text();needle="iw=W/f'intra_psnr_TMM_q{q+1}.pth.tar';pw=W/f'inter_psnr_TMM_q{q+1}.pth'";assert source.count(needle)==1
mapping={q-1:str(selected[q]) for q in selected};hashes={q-1:sha(selected[q]) for q in selected}
source=source.replace(needle,"iw=W/f'intra_psnr_TMM_q{q+1}.pth.tar';pw=Path("+repr(mapping)+"[q]);assert sha(pw)=="+repr(hashes)+"[q]")
compile(source,str(target/'code/learned_worker.py'),'exec');(target/'code/learned_worker.py').write_text(source)
for n in ['input_manifest.json','source_sha256.json']:shutil.copy2(BASE/'protocol'/n,target/'protocol'/n)
for n in ['rt_metrics.py','rt_transforms.py']:shutil.copy2(BASE/'references'/n,target/'references'/n)
cfg=read(BASE/'protocol/benchmark.json');cfg.update(training_arm=a.arm,selected_p_sha256={str(k):v for k,v in hashes.items()});path=target/'protocol/benchmark.json'
if path.exists():assert read(path)==cfg
else:atomic(path,cfg)
if not (target/'inputs/yuv420').exists():(target/'inputs/yuv420').symlink_to(BASE/'inputs/yuv420',target_is_directory=True)
roots={};reuse=[]
for q in [1,2,3,4]:
    if not selections[q]['accepted']:
        x=state(selected[q]);y=state(WEIGHTS/f'inter_psnr_TMM_q{q}.pth');assert x.keys()==y.keys() and all(x[k].equal(y[k]) for k in x);roots[q]=BASE;reuse.append(q)
    else:
        cmd=[PY,'-B',str(target/'code/learned_worker.py'),'--method','tcm','--gpu',str(a.gpu),'--rates',str(q-1)]
        with (target/f'q{q}.log').open('a') as f:
            subprocess.run(cmd+['--pilot'],cwd=target,stdout=f,stderr=subprocess.STDOUT,check=True)
            first=read(BASE/'protocol/input_manifest.json')['clips'][0]['clip'];d=read(target/f'results/tcm/q{q-1}/{first}.json');assert d['independent_decode_checked']
            atomic(target/f'protocol/q{q}_pilot_audit.json',dict(passed=True,fresh_independent_decode=True,frames=96,p_sha256=hashes[q-1]))
            subprocess.run(cmd,cwd=target,stdout=f,stderr=subprocess.STDOUT,check=True)
        roots[q]=target
curves=[];result_files=[];stream_count=0
for q in [1,2,3,4]:
    by=collections.defaultdict(list)
    for c in read(BASE/'protocol/input_manifest.json')['clips']:
        path=roots[q]/f'results/tcm/q{q-1}/{c["clip"]}.json';d=read(path)
        assert d['input_sha256']==c['sha256'] and d['frames']==len(d['per_frame'])==96 and d['i_positions']==[0,32,64] and d['true_bitstream']
        assert sum(s['bytes'] for s in d['streams'])==d['bitstream_bytes'] and all(sha(s['path'])==s['sha256'] for s in d['streams'])
        stream_count+=len(d['streams']);by[c['camera']].append(d);result_files.append(dict(quality=q,clip=c['clip'],path=str(path),sha256=sha(path),reused_original=q in reuse))
    assert len(by)==82 and all(len(group)==3 for group in by.values())
    curves.append(dict(quality=q,**{k:sum(sum(d[k] for d in group)/3 for group in by.values())/82 for k in ['bpp','psnr','psnr_y','psnr_u','psnr_v','ms_ssim']}))
assert len(result_files)==984 and stream_count==94464
atomic(target/'result_index.json',dict(results=result_files))
atomic(target/'rd_summary.json',dict(arm=a.arm,curves=curves,aggregation='82 original videos equally weighted, three96frame windows/video',original_quality_reused=reuse,selection=selections))
atomic(target/'complete_audit.json',dict(passed=True,results=984,frames=94464,streams=stream_count,all_stream_hashes_verified=True,original_tensor_identity_quality_reuse=reuse,test_used_for_selection=False))
