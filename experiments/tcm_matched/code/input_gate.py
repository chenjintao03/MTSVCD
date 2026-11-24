"""CPU-only format/identity checks. This does not validate or train TCM models."""
from common import *
setup()
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import numpy as np
import subprocess,traceback
from data import self_ids,self_frames,vimeo_frames,vimeo_ids,Samples,colour,META
torch.set_num_threads(2)
report=dict(status='running',gpu_used=False,checks=[])
def save():atomic(ROOT/'diagnostic/input_gate.json',report)
save()
try:
    assert not torch.cuda.is_available()
    splits={s:self_ids(s) for s in ['train','val','test']}
    assert [len(splits[s]) for s in splits]==[380,81,82]
    groups={s:{r['reviewed_group_id'] for r in rows} for s,rows in splits.items()}
    assert all(not groups[a]&groups[b] for a,b in [('train','val'),('train','test'),('val','test')])
    vt,vv=vimeo_ids('train'),vimeo_ids('val');assert len(vt)==64612 and len(vv)==7824 and not set(vt)&set(vv)
    frozen=read(RT_ROOT/'diagnostic/input_audit.json');assert frozen['status']=='passed'
    for item in frozen['checks']:
        assert item['split'] in ['train','val'];r=next(r for r in splits[item['split']] if r['camera']==item['camera']);start=item['start'];src=Path('@DATASET_ROOT@/processed_video')/r['camera']/'clip01.mp4'
        assert sha(src)==item['source_sha256']
        got=self_frames(r,start,5)
        raw=subprocess.check_output(['/usr/bin/ffmpeg','-v','error','-threads','1','-i',str(src),'-vf',f'select=between(n\\,{start}\\,{start+4})','-frames:v','5','-vsync','0','-c:v','rawvideo','-threads','1','-pix_fmt',META[r['camera']]['pix_fmt'],'-f','rawvideo','-'])
        frames=np.frombuffer(raw,np.uint8).reshape(5,368640);refs=[]
        for a in frames:
            y=a[:245760].reshape(1,384,640);uv=np.stack([a[245760:307200].reshape(192,320),a[307200:].reshape(192,320)])
            t=torch.from_numpy(np.concatenate([y,uv.repeat(2,1).repeat(2,2)],0).copy()).float()[None]/255
            refs.append(torch.round(colour.ycbcr2rgb(t)*255).clamp(0,255)[0]/255)
        expected=torch.stack(refs);assert torch.equal(got,expected)
        report['checks'].append(dict(camera=r['camera'],split=item['split'],frames=5,source_unchanged=True,canonical_rgb8_matches_formal_adapter=True,frame_index_decode_matches_seek=True,normalized_float_pixels_sha256=hashlib.sha256(got.numpy().tobytes()).hexdigest()));save()
    for split,ids in [('train',vt),('val',vv)]:
        for cid in ids[:4]:
            frames=vimeo_frames(cid,1,5);assert frames.shape==(5,3,256,448) and torch.isfinite(frames).all() and frames.min()>=0 and frames.max()<=1
            report['checks'].append(dict(vimeo_sequence=cid,split=split,frames=5,png_indices=[2,3,4,5,6],canonical_projection='RGB->BT709->420->RGB8',pixels_sha256=hashlib.sha256(frames.numpy().tobytes()).hexdigest()));save()
    for arm in ['surveillance','vimeo']:
        ds=Samples(arm,1);assert len(ds)==12000 and torch.equal(ds[0],ds[0])
        for n in range(4):assert ds[n].shape==(5,3,256,256)
    report.update(status='passed',test_frames_used=False,source_transform_sha256=sha(ROOT/'references/rt_transforms.py'),effective_batch_planned=4,model_gradient_and_real_stream_gates_still_required=True)
    save()
except BaseException:report.update(status='failed',error=traceback.format_exc());save();raise
