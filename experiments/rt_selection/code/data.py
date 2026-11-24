"""Existing RT data recipe plus explicit external-source readers; no test-driven selection."""
from pathlib import Path
import importlib.util,random,subprocess
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from common import ROOT,read
from src.utils.transforms import rgb2ycbcr
LEGACY=Path('@WORKSPACE@/rt_matched')
spec=importlib.util.spec_from_file_location('frozen_rt_data',LEGACY/'code/data.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
Validation=legacy.Validation;ThreadBatches=legacy.ThreadBatches
def external_frames(record,start,count=7,fast=False):
    if record['kind']=='avi':
        ff=['/usr/bin/ffmpeg','-v','error','-threads','1']
        if fast:ff+=['-ss',f'{start/record["fps"]:.12f}','-i',record['path']]
        else:ff+=['-i',record['path'],'-vf',f'select=between(n\\,{start}\\,{start+count-1})']
        ff+=['-frames:v',str(count),'-vsync','0','-c:v','rawvideo','-threads','1','-pix_fmt','yuv420p','-f','rawvideo','-']
        b=subprocess.check_output(ff);h,w=record['height'],record['width'];size=h*w
        a=np.frombuffer(b,np.uint8).reshape(count,size*3//2)
        y=a[:,:size].reshape(count,1,h,w);u=a[:,size:size*5//4].reshape(count,1,h//2,w//2);v=a[:,size*5//4:].reshape(count,1,h//2,w//2)
        uv=torch.from_numpy(np.concatenate([u,v],1).copy()).repeat_interleave(2,2).repeat_interleave(2,3)
        return torch.cat([torch.from_numpy(y.copy()),uv],1).float()/255
    assert record['kind']=='jpeg'
    paths=record['frames'][start:start+count];assert len(paths)==count
    arr=np.stack([np.asarray(Image.open(p).convert('RGB')).copy() for p in paths])
    yuv=rgb2ycbcr(torch.from_numpy(arr).permute(0,3,1,2).float()/255)
    uv=F.avg_pool2d(yuv[:,1:],2,2).repeat_interleave(2,2).repeat_interleave(2,3)
    return torch.cat([yuv[:,:1],uv],1)
class Clips:
    def __init__(self,arm,split,steps=3000,effective_batch=16,seed=20261104):
        assert arm in ['shanghai','ua_detrac'] and split=='train'
        self.ids=read(ROOT/f'protocol/{arm}_training_manifest.json')['sequences'];self.seed=seed;self.count=steps*effective_batch
        self.order=list(range(len(self.ids)));random.Random(seed).shuffle(self.order)
    def __len__(self):return self.count
    def __getitem__(self,index):
        rng=random.Random(self.seed+index*1000003);r=self.ids[self.order[index%len(self.ids)]];start=rng.randrange(r['frame_count']-6)
        return legacy.crop(external_frames(r,start),rng)
