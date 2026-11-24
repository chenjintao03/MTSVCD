from pathlib import Path
import os,sys,json,time,hashlib
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('@WORKSPACE@/rt_original')
BASE=Path('@WORKSPACE@/native420_benchmark')
TCM=Path('@WORKSPACE@/tcm_matched')
PY='@RT_PYTHON@'
UUID={4:'@GPU4_UUID@',5:'@GPU5_UUID@'}
WEIGHTS={'image':'555eff5f4026774f477bebdcbb3b52548e0da230803959dcebcea4d732a90dd9','video':'b12e7faf4ddb6126d8e138a627ed6a349b8e1052d3ed9e343e1ba266466675d6'}
def setup(gpu=None):
    for key,sub in dict(TMP='tmp',TEMP='tmp',TMPDIR='tmp',XDG_CACHE_HOME='cache',CUDA_CACHE_PATH='cache/cuda',TORCH_HOME='cache/torch',MPLCONFIGDIR='cache/matplotlib',TRITON_CACHE_DIR='cache/triton',TORCHINDUCTOR_CACHE_DIR='cache/inductor').items():
        p=ROOT/sub;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if gpu is not None:assert gpu in UUID
    os.environ['CUDA_VISIBLE_DEVICES']=UUID[gpu] if gpu is not None else ''
    os.chdir(ROOT);sys.path[:0]=[str(OLD/'vendor'),str(OLD/'implementation')]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8388608),b''):h.update(block)
    return h.hexdigest()
def atomic(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    for n in range(100):
        try:tmp.replace(p);return
        except PermissionError:
            if n==99:raise
            time.sleep(.1)
def source_identity():
    cfg=read(ROOT/'protocol/experiment.json');assert all(sha(ROOT/path)==value for path,value in cfg['source_sha256'].items())
    assert sha(ROOT/'protocol/metadata_and_windows.json')==cfg['metadata_sha256'],'Frozen native source/window metadata changed'
    original=read(BASE/'protocol/source_sha256.json')['rt'];assert all(sha(p)==value for p,value in original.items())
    assert all(sha(OLD/'checkpoints'/f'cvpr2025_{name}.pth.tar')==value for name,value in WEIGHTS.items())
    return original
