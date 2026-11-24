"""Canonical RGB8 inputs matching the frozen native420 TCM benchmark adapter.

Do not import DCVC-RT's src package in a process importing the TCM src package.
Only its standalone, SHA-frozen transform file is imported by file path.
"""
from common import ROOT,read
import importlib.util,random,subprocess,os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
spec=importlib.util.spec_from_file_location('tcm_canonical_colour',ROOT/'references/rt_transforms.py');colour=importlib.util.module_from_spec(spec);spec.loader.exec_module(colour)
SPLIT=read(ROOT/'protocol/self_split.json');META=read(ROOT/'protocol/self_metadata.json')
V=Path('@VIMEO_ROOT@')
def self_ids(split):return [r for r in SPLIT['records'] if r['subset']=='short' and r['split']==split]
def vimeo_ids(split):return [s.strip() for s in (ROOT/'protocol'/('sep_trainlist.txt' if split=='train' else 'sep_testlist.txt')).read_text().splitlines() if s.strip()]
def native_to_rgb(yuv):
    # The frozen benchmark converts on GPU. CPU and CUDA arithmetic can round
    # differently near half-integer RGB8 thresholds, despite identical formulas.
    # Reproduce the benchmark device before cropping; then stage data on CPU.
    device=os.environ.get('TCM_CANONICAL_RGB8_DEVICE','cpu')
    return (torch.round(colour.ycbcr2rgb(yuv.to(device))*255).clamp(0,255)/255).cpu()
def self_frames(record,start,count):
    src=Path('@DATASET_ROOT@/processed_video')/record['camera']/'clip01.mp4';pix=META[record['camera']]['pix_fmt']
    b=subprocess.check_output(['/usr/bin/ffmpeg','-v','error','-threads','1','-ss',f'{start/30:.9f}','-i',str(src),'-frames:v',str(count),'-vsync','0','-c:v','rawvideo','-threads','1','-pix_fmt',pix,'-f','rawvideo','-'])
    a=np.frombuffer(b,np.uint8).reshape(count,368640);y=a[:,:245760].reshape(count,1,384,640);uv=np.stack([a[:,245760:307200].reshape(count,192,320),a[:,307200:].reshape(count,192,320)],1)
    yuv=torch.cat([torch.from_numpy(y.copy()),torch.from_numpy(uv.copy()).repeat_interleave(2,2).repeat_interleave(2,3)],1).float()/255
    return native_to_rgb(yuv)
def vimeo_frames(cid,start=1,count=5):
    # start is zero-based among the seven official PNG frames.
    t=torch.from_numpy(np.stack([np.asarray(Image.open(V/'sequences'/cid/f'im{n+1}.png').convert('RGB')).copy() for n in range(start,start+count)])).permute(0,3,1,2).float()/255
    yuv=colour.rgb2ycbcr(t);uv=F.avg_pool2d(yuv[:,1:],2,2).repeat_interleave(2,2).repeat_interleave(2,3)
    return native_to_rgb(torch.cat([yuv[:,:1],uv],1))
class Samples:
    def __init__(self,arm,quality,updates=3000,offset=0):
        self.arm=arm;self.ids=self_ids('train') if arm=='surveillance' else vimeo_ids('train');self.seed=20261105+quality;self.length=updates*4;self.offset=offset;self.orders={}
    def __len__(self):return self.length-self.offset
    def __getitem__(self,index):
        index+=self.offset;epoch,pos=divmod(index,len(self.ids))
        if epoch not in self.orders:
            order=list(range(len(self.ids)));random.Random(self.seed+epoch).shuffle(order);self.orders[epoch]=order
        item=self.ids[self.orders[epoch][pos]];rng=random.Random(self.seed+index*1000003)
        if self.arm=='surveillance':t=self_frames(item,rng.randrange(META[item['camera']]['frames']-4),5)
        else:t=vimeo_frames(item,rng.randrange(3),5)
        h,w=t.shape[-2:];y=2*rng.randrange((h-256)//2+1);x=2*rng.randrange((w-256)//2+1);t=t[:,:,y:y+256,x:x+256]
        if rng.random()<.5:t=t.flip(-1)
        return t.contiguous()
class Validation:
    def __init__(self,domain='surveillance',limit=None):
        self.domain=domain;self.ids=self_ids('val') if domain=='surveillance' else vimeo_ids('val')
        if limit:
            self.ids=list(self.ids);random.Random(20261105).shuffle(self.ids);self.ids=self.ids[:limit]
    def __len__(self):return len(self.ids)*3 if self.domain=='surveillance' else len(self.ids)
    def __getitem__(self,index):
        if self.domain=='vimeo':return vimeo_frames(self.ids[index],1,5)
        r=self.ids[index//3];n=META[r['camera']]['frames'];start=[0,(n-96)//2,n-96][index%3]
        return self_frames(r,start,96)
class ThreadBatches:
    def __init__(self,dataset,batch_size=1,workers=2,prefetch=4):self.ds=dataset;self.bs=batch_size;self.workers=workers;self.prefetch=prefetch
    def __iter__(self):
        from concurrent.futures import ThreadPoolExecutor
        from collections import deque
        def batch(start):return torch.stack([self.ds[n] for n in range(start,min(start+self.bs,len(self.ds)))])
        indices=iter(range(0,len(self.ds),self.bs))
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            pending=deque()
            for _ in range(self.prefetch):
                try:pending.append(ex.submit(batch,next(indices)))
                except StopIteration:break
            while pending:
                value=pending.popleft().result()
                try:pending.append(ex.submit(batch,next(indices)))
                except StopIteration:pass
                yield value
