from common import *
setup()
import collections,subprocess,random
import numpy as np
from data import self_ids,self_frames,vimeo_ids
report=dict(status='running',updated=time.strftime('%Y-%m-%d %H:%M:%S'),checks=[])
def save():atomic(ROOT/'diagnostic/input_audit.json',report)
save()
try:
    splits={s:self_ids(s) for s in ['train','val','test']}
    assert [len(splits[s]) for s in splits]==[380,81,82]
    groups={s:{r['reviewed_group_id'] for r in rs} for s,rs in splits.items()}
    for a,b in [('train','val'),('train','test'),('val','test')]:assert not groups[a]&groups[b]
    vt, vv=vimeo_ids('train'),vimeo_ids('val')
    assert len(vt)==len(set(vt))==64612 and len(vv)==len(set(vv))==7824 and not set(vt)&set(vv)
    chosen=[]
    for split in ['train','val']:
        # Round-robin categories; no test frames or RD ranking used.
        buckets=collections.defaultdict(list)
        for r in sorted(splits[split],key=lambda x:x['camera']):buckets[r['effective_category']].append(r)
        selected=[]
        while len(selected)<8:
            for k in sorted(buckets):
                if buckets[k] and len(selected)<8:selected.append(buckets[k].pop(0))
        for record in selected:
            src=Path('@DATASET_ROOT@/processed_video')/record['camera']/'clip01.mp4'
            probe=json.loads(subprocess.check_output(['/usr/bin/ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(src)]))['streams'][0]
            assert (probe['width'],probe['height'])==(640,384) and probe['pix_fmt'] in ['yuv420p','yuvj420p']
            assert probe['r_frame_rate']=='30/1'
            meta=read(ROOT/'protocol/self_metadata.json')[record['camera']]
            assert int(probe['nb_frames'])==meta['frames']
            start=(meta['frames']-7)//2
            quick=self_frames(record,start).numpy()
            raw=subprocess.check_output(['/usr/bin/ffmpeg','-v','error','-threads','1','-i',str(src),'-vf',f'select=between(n\\,{start}\\,{start+6})','-frames:v','7','-vsync','0','-c:v','rawvideo','-threads','1','-pix_fmt',probe['pix_fmt'],'-f','rawvideo','-'])
            a=np.frombuffer(raw,np.uint8).reshape(7,-1);y=a[:,:245760].reshape(7,1,384,640);uv=np.stack([a[:,245760:307200].reshape(7,192,320),a[:,307200:].reshape(7,192,320)],1)
            exact=np.concatenate([y,uv.repeat(2,2).repeat(2,3)],1).astype(np.float32)/255
            assert np.array_equal(quick,exact),(record['camera'],'seek/frame-index mismatch')
            chosen.append(dict(camera=record['camera'],split=split,category=record['effective_category'],start=start,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),pixels_sha256=hashlib.sha256(quick.tobytes()).hexdigest(),metadata={k:probe.get(k) for k in ['codec_name','width','height','pix_fmt','color_range','color_space','r_frame_rate','nb_frames']},seek_equals_frame_index=True))
            report['checks']=chosen;save()
    report.update(status='passed',short_split_counts={s:len(x) for s,x in splits.items()},no_group_leakage=True,vimeo_official_list_counts=[len(vt),len(vv)],test_frames_used=False,limitations=['Historical direct resize to640x384 was user-confirmed; interpolation kernel and original capture provenance remain unverified.','This audit checks current MP4 decoding/frame correspondence; it cannot prove absence of artifacts already present in the source or prior resizing.'])
    save()
except BaseException:
    import traceback
    report.update(status='failed',error=traceback.format_exc());save();raise
