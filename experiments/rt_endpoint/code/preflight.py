from common import *
setup()
import subprocess,concurrent.futures,re,random,traceback
from PIL import Image
from fractions import Fraction
OLD=Path('@WORKSPACE@/rt_matched')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def probe(p):
    d=json.loads(subprocess.check_output(['/usr/bin/ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,avg_frame_rate,pix_fmt,codec_name','-of','json',str(p)],text=True))['streams'][0]
    frames=int(d['nb_frames']);fps=float(Fraction(d['avg_frame_rate']));w,h=d['width'],d['height'];assert frames>=7 and min(w,h)>=256 and w%2==h%2==0 and fps>0
    return dict(id=p.stem,kind='avi',path=str(p),sha256=sha(p),bytes=p.stat().st_size,width=w,height=h,fps=fps,frame_count=frames,codec=d['codec_name'],source_pix_fmt=d['pix_fmt'])
try:
    assert read(OLD/'complete_audit.json')['passed']
    copy_audit=read(ROOT/'shanghai_copy_audit.json');assert copy_audit['passed'] and copy_audit['files']==330
    paths=sorted(Path('@DATASET_ROOT@/ShanghaiTech/training/videos').glob('*.avi'));assert len(paths)==330
    expected={r['name']:r for r in read(ROOT/'shanghai_copy_manifest.json')}
    cache=read(ROOT/'protocol/source_manifest_cache_audit.json') if (ROOT/'protocol/source_manifest_cache_audit.json').exists() else None
    if cache:
        assert sha(ROOT/'protocol/shanghai_training_manifest.json')==cache['shanghai_manifest_sha256']
        videos=read(ROOT/'protocol/shanghai_training_manifest.json')['sequences']
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:videos=list(ex.map(probe,paths))
    assert all(v['sha256']==expected[Path(v['path']).name]['sha256'] for v in videos)
    atomic(ROOT/'protocol/shanghai_training_manifest.json',dict(sequences=videos,official_train_only=True,official_test_used=False))
    ua=Path('@DATASET_ROOT@/UA-DETRAC/Insight-MVT_Annotation_Train');sequences=[]
    if cache:
        assert sha(ROOT/'protocol/ua_detrac_training_manifest.json')==cache['ua_manifest_sha256']
        sequences=read(ROOT/'protocol/ua_detrac_training_manifest.json')['sequences']
    for folder in ([] if cache else sorted(ua.iterdir())):
        if not folder.is_dir():continue
        files=sorted(folder.glob('*.jpg'));assert len(files)>=7
        numbers=[int(re.search(r'(\d+)\.jpg$',p.name,re.I)[1]) for p in files];assert numbers==list(range(numbers[0],numbers[0]+len(numbers)))
        with Image.open(files[0]) as im:w,h=im.size
        assert min(w,h)>=256 and w%2==h%2==0
        hashes={str(p):sha(p) for p in files}
        sequences.append(dict(id=folder.name,kind='jpeg',path=str(folder),frames=[str(p) for p in files],frame_sha256=hashes,frame_count=len(files),width=w,height=h,fps=25,fps_source='official_nominal_not_measured_clock'))
    assert len(sequences)==60
    atomic(ROOT/'protocol/ua_detrac_training_manifest.json',dict(sequences=sequences,official_train_only=True,official_test_used=False))
    from data import external_frames,Clips
    import torch
    torch.set_num_threads(2)
    inputs=[]
    for arm,rs in [('shanghai',videos),('ua_detrac',sequences)]:
        for r in rs[:2]:
            for start in [0,(r['frame_count']-7)//2]:
                fast=external_frames(r,start,fast=False);assert torch.isfinite(fast).all() and fast.shape==(7,3,r['height'],r['width'])
                if arm=='shanghai':
                    import numpy as np
                    h,w=r['height'],r['width'];size=h*w
                    raw=subprocess.check_output(['/usr/bin/ffmpeg','-v','error','-threads','1','-i',r['path'],'-frames:v',str(start+7),'-vsync','0','-c:v','rawvideo','-threads','1','-pix_fmt','yuv420p','-f','rawvideo','-'])
                    a=np.frombuffer(raw,np.uint8).reshape(start+7,size*3//2)[start:]
                    y=torch.from_numpy(a[:,:size].copy()).reshape(7,1,h,w);u=a[:,size:size*5//4].reshape(7,1,h//2,w//2);v=a[:,size*5//4:].reshape(7,1,h//2,w//2)
                    uv=torch.from_numpy(np.concatenate([u,v],1).copy()).repeat_interleave(2,2).repeat_interleave(2,3)
                    reference=torch.cat([y,uv],1).float()/255
                    assert torch.equal(fast,reference),'Exact frame filter vs independent sequential-prefix decode mismatch'
                inputs.append(dict(arm=arm,sequence=r['id'],start=start,pixels_sha256=hashlib.sha256(fast.numpy().tobytes()).hexdigest(),exact_frame_filter_vs_independent_sequential_prefix=arm=='shanghai'))
        sample=Clips(arm,'train',steps=1)[0];assert sample.shape==(7,3,256,256)
    # Existing training arms are fixed at update3000, not the old selected/original fallback.
    import torch
    reused={}
    for arm in ['vimeo','surveillance']:
        candidate=OLD/arm/'candidate_step3000.pth.tar';last=OLD/arm/'last.pth.tar'
        d=torch.load(candidate,map_location='cpu',weights_only=False);c=torch.load(last,map_location='cpu',weights_only=False)
        assert d['updates']==c['updates']==3000 and d['sampler_offset']==c['sampler_offset']==48000
        assert d['protocol']==c['protocol']==read(OLD/'protocol/experiment.json')
        assert d['state_dict'].keys()==c['state_dict'].keys() and all(torch.equal(v,c['state_dict'][k]) for k,v in d['state_dict'].items())
        (ROOT/arm).mkdir(exist_ok=True);target=ROOT/arm/'endpoint.pth.tar'
        if target.exists():
            existing=state(target);assert all(torch.equal(v,existing[k]) for k,v in d['state_dict'].items())
        else:torch.save(dict(state_dict=d['state_dict']),target)
        reused[arm]=dict(passed=True,updates=3000,sampler_offset=48000,source_checkpoint=str(candidate),source_sha256=sha(candidate),endpoint_sha256=sha(target),final_step_tensor_identity=True)
        if arm=='surveillance':
            existing=state(OLD/arm/'selected.pth.tar');assert all(torch.equal(v,existing[k]) for k,v in d['state_dict'].items())
            assert read(OLD/'evaluation/surveillance/complete_audit.json')['passed']
    atomic(ROOT/'protocol/reused_endpoint_audit.json',reused)
    atomic(ROOT/'diagnostic/input_audit.json',dict(status='passed',shanghai_sequences=330,ua_sequences=60,checks=inputs,source_dataset_test_not_read=True,new_target_test_not_read=True))
    print('INPUT_PREFLIGHT_PASSED',flush=True)
except BaseException:atomic(ROOT/'diagnostic/input_audit.json',dict(status='failed',error=traceback.format_exc()));raise
