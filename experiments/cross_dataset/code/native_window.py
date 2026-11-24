"""Offset/count wrapper around the unmodified official native420 reader."""
from pathlib import Path
import hashlib
class NativeWindow:
    def __init__(self,clip,reader_class):
        self.clip=clip;self.done=0;self.closed=False;self.hasher=hashlib.sha256()
        width,height=clip['width'],clip['height'];assert width%2==height%2==0
        assert clip['frame_bytes']==width*height*3//2 and clip['frames']==96
        assert clip['start']>=0 and clip['start']+96<=clip['source_official_frames']
        assert Path(clip['source_path']).stat().st_size==clip['source_bytes']
        self.reader=reader_class(clip['source_path'],width,height)
        self.reader.file.seek(clip['offset_bytes']);assert self.reader.file.tell()==clip['start']*clip['frame_bytes']
    def read_one_frame(self):
        if self.done>=96:raise EOFError('Reading beyond the frozen96frame window is forbidden')
        y,uv=self.reader.read_one_frame();c=self.clip
        assert y is not None and uv is not None and y.dtype.name=='uint8' and uv.dtype.name=='uint8'
        assert y.shape==(1,c['height'],c['width']) and uv.shape==(2,c['height']//2,c['width']//2)
        self.hasher.update(y.tobytes());self.hasher.update(uv.tobytes());self.done+=1;return y,uv
    def close(self):
        if not self.closed:self.reader.close();self.closed=True
        assert self.done==96,'Official encode/decode did not consume exactly96 reference frames'
        if 'sha256' in self.clip:assert self.hasher.hexdigest()==self.clip['sha256'],'Reference window changed'
    def evidence(self):return dict(frames=self.done,window_sha256=self.hasher.hexdigest(),closed=self.closed)
