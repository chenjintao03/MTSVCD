from pathlib import Path
import os,sys,json,hashlib,time
ROOT=Path(__file__).resolve().parents[1]
def setup(gpu=None):
 for key,sub in dict(TMPDIR='tmp',TMP='tmp',TEMP='tmp',XDG_CACHE_HOME='cache',CUDA_CACHE_PATH='cache/cuda',TORCH_HOME='cache/torch',PIP_CACHE_DIR='cache/pip',MPLCONFIGDIR='cache/matplotlib',TORCHINDUCTOR_CACHE_DIR='cache/inductor',TRITON_CACHE_DIR='cache/triton').items():
  p=ROOT/sub;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
 os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 if gpu is not None:
  assert gpu in [4,5];os.environ['CUDA_VISIBLE_DEVICES']={4:'@GPU4_UUID@',5:'@GPU5_UUID@'}[gpu]
 os.chdir(ROOT);sys.path[:0]=[str(ROOT/'implementation'),'@WORKSPACE@/rt_original/vendor']
def atomic(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8')
 for attempt in range(100):
  try:t.replace(p);return
  except PermissionError:
   if attempt==99:raise
   time.sleep(.1)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def state(p):
 import torch
 d=torch.load(p,map_location='cpu',weights_only=False);d=d.get('state_dict',d);return {k.removeprefix('module.'):v for k,v in d.items()}
I=Path('@WORKSPACE@/rt_original/checkpoints/cvpr2025_image.pth.tar')
P=I.with_name('cvpr2025_video.pth.tar')
