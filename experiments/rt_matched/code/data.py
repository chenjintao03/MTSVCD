from pathlib import Path
import random,subprocess
from common import ROOT,read
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from src.utils.transforms import rgb2ycbcr
V=Path('@VIMEO_ROOT@')
def vimeo_ids(split):return [s.strip() for s in (V/('sep_trainlist.txt' if split=='train' else 'sep_testlist.txt')).read_text().splitlines() if s.strip()]
def self_ids(split):return [r for r in read(ROOT/'protocol/self_split.json')['records'] if r.get('subset')=='short' and r.get('split')==split]
def self_frames(record,start,count=7):
 src=Path('@DATASET_ROOT@/processed_video')/record['camera']/'clip01.mp4'
 pix=read(ROOT/'protocol/self_metadata.json')[record['camera']]['pix_fmt']
 b=subprocess.check_output(['/usr/bin/ffmpeg','-v','error','-threads','1','-ss',f'{start/30:.9f}','-i',str(src),'-frames:v',str(count),'-vsync','0','-c:v','rawvideo','-threads','1','-pix_fmt',pix,'-f','rawvideo','-'])
 a=np.frombuffer(b,dtype=np.uint8).reshape(count,-1);y=a[:,:245760].reshape(count,1,384,640);uv=np.stack([a[:,245760:307200].reshape(count,192,320),a[:,307200:].reshape(count,192,320)],1)
 return torch.cat([torch.from_numpy(y.copy()),torch.from_numpy(uv.copy()).repeat_interleave(2,2).repeat_interleave(2,3)],1).float()/255
def vimeo_frames(cid):
 a=np.stack([np.asarray(Image.open(V/'sequences'/cid/f'im{n}.png').convert('RGB')) for n in range(1,8)])
 t=rgb2ycbcr(torch.from_numpy(a.copy()).permute(0,3,1,2).float()/255)
 return torch.cat([t[:,:1],F.avg_pool2d(t[:,1:],2,2).repeat_interleave(2,2).repeat_interleave(2,3)],1)
def crop(t,rng=None):
 h,w=t.shape[-2:]
 if rng is None:y=(h-256)//2;x=(w-256)//2
 else:y=2*rng.randrange((h-256)//2+1);x=2*rng.randrange((w-256)//2+1)
 t=t[:,:,y:y+256,x:x+256]
 if rng is not None and rng.random()<.5:t=t.flip(-1)
 if rng is not None and rng.random()<.5:t=t.flip(0)
 return t.contiguous()
class Clips(torch.utils.data.Dataset):
 def __init__(self,arm,split,steps=3000,effective_batch=16,seed=20261103):
  self.arm=arm;self.split=split;self.seed=seed;self.ids=self_ids(split) if arm=='surveillance' else vimeo_ids(split)
  self.count=steps*effective_batch if split=='train' else len(self.ids)
  self.order=list(range(len(self.ids)));random.Random(seed).shuffle(self.order)
 def __len__(self):return self.count
 def __getitem__(self,index):
  rng=random.Random(self.seed+index*1000003);item=self.ids[self.order[index%len(self.ids)]]
  if self.arm=='vimeo':t=vimeo_frames(item)
  else:
   frames=read(ROOT/'protocol/self_metadata.json')[item['camera']]['frames'];start=rng.randrange(frames-6) if self.split=='train' else (frames-7)//2;t=self_frames(item,start)
  return crop(t,rng if self.split=='train' else None)

class Validation(torch.utils.data.Dataset):
 def __init__(self,arm,limit=None):
  self.arm=arm;self.ids=self_ids('val') if arm=='surveillance' else vimeo_ids('val')
  if limit:
   random.Random(20261103).shuffle(self.ids);self.ids=self.ids[:limit]
 def __len__(self):return len(self.ids)*3 if self.arm=='surveillance' else len(self.ids)
 def __getitem__(self,index):
  if self.arm=='vimeo':return vimeo_frames(self.ids[index])
  item=self.ids[index//3];n=read(ROOT/'protocol/self_metadata.json')[item['camera']]['frames'];start=[0,(n-96)//2,n-96][index%3]
  return self_frames(item,start,96)

class ThreadBatches:
 """Bounded in-memory prefetch without multiprocessing Unix sockets on DrvFS."""
 def __init__(self,dataset,batch_size,workers=2,prefetch=4):self.ds=dataset;self.bs=batch_size;self.workers=workers;self.prefetch=prefetch
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
