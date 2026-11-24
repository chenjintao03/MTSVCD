from pathlib import Path
import os,sys,json,time,hashlib
ROOT=Path(__file__).resolve().parents[1]
PY='@TCM_PYTHON@'
RT_ROOT=Path('@WORKSPACE@/rt_matched')
BASE=Path('@WORKSPACE@/native420_benchmark')
WEIGHTS=Path('@TCM_WEIGHTS@')
UUID={4:'@GPU4_UUID@',5:'@GPU5_UUID@'}
LAMBDAS={1:256,2:512,3:1024,4:2048}
def setup(gpu=None):
    for k,sub in dict(TMPDIR='tmp',TMP='tmp',TEMP='tmp',XDG_CACHE_HOME='cache',CUDA_CACHE_PATH='cache/cuda',TORCH_HOME='cache/torch',MPLCONFIGDIR='cache/matplotlib',TRITON_CACHE_DIR='cache/triton',TORCHINDUCTOR_CACHE_DIR='cache/inductor').items():
        p=ROOT/sub;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if gpu is not None:
        assert gpu in UUID;os.environ['CUDA_VISIBLE_DEVICES']=UUID[gpu]
        os.environ['TCM_CANONICAL_RGB8_DEVICE']='cuda:0'
    else:os.environ['TCM_CANONICAL_RGB8_DEVICE']='cpu'
    os.chdir(ROOT);sys.path.insert(0,str(ROOT/'implementation'))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def atomic(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(d,indent=2),encoding='utf-8')
    for n in range(100):
        try:t.replace(p);return
        except PermissionError:
            if n==99:raise
            time.sleep(.1)
def state(p):
    import torch
    d=torch.load(p,map_location='cpu',weights_only=False);return {k.removeprefix('module.'):v for k,v in d.get('state_dict',d).items()}
