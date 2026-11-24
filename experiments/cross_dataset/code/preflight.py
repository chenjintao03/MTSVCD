"""CPU-only source and all48native-window identity audit; no encoding."""
from common import *
setup()
os.nice(10)
import ast,hashlib,numpy as np
from native_window import NativeWindow
from src.utils.video_reader import YUV420Reader
runtime=source_identity();metadata=read(ROOT/'protocol/metadata_and_windows.json');assert len(metadata['clips'])==48
sources={c['source_path']:c for c in metadata['clips']};assert len(sources)==16
for path,c in sources.items():assert Path(path).stat().st_size==c['source_bytes'] and sha(path)==c['source_sha256']
clips=[];checks=[]
for item in metadata['clips']:
    c=dict(item);reader=NativeWindow(c,YUV420Reader);first=last=None
    for n in range(96):
        y,uv=reader.read_one_frame()
        if n==0:first=y.tobytes()+uv.tobytes()
        if n==95:last=y.tobytes()+uv.tobytes()
    reader.close()
    with Path(c['source_path']).open('rb') as f:
        f.seek(c['offset_bytes']);assert first==f.read(c['frame_bytes'])
        f.seek(c['offset_bytes']+95*c['frame_bytes']);assert last==f.read(c['frame_bytes'])
    c['sha256']=reader.hasher.hexdigest();clips.append(c);checks.append(dict(clip=c['clip'],first_last_exact_offsets=True,reader_frames=reader.done,sha256=c['sha256']))
for path in (ROOT/'code').glob('*.py'):ast.parse(path.read_text())
atomic(ROOT/'protocol/input_manifest.json',dict(clips=clips,source_metadata_sha256=sha(ROOT/'protocol/metadata_and_windows.json'),raw_sources=16,reader='unmodified officialYUV420Reader with checked seek and96frame limit'))
atomic(ROOT/'protocol/runtime_sources.json',runtime)
atomic(ROOT/'diagnostic/cpu_input_gate.json',dict(passed=True,cpu_only=True,gpu_used=False,training_started=False,encoding_started=False,sources=16,windows=48,frames=4608,original_reader_sha256=sha(OLD/'implementation/src/utils/video_reader.py'),all_source_hashes_verified=True,all_first_last_frame_offsets_exact=True,checks=checks))
print('CPU_NATIVE420_WINDOWS_PASSED48; noGPU or encoding',flush=True)
