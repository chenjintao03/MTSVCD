from common import *
setup()
import torch
cfg=read(ROOT/'protocol/experiment.json')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
assert cfg['no_training'] and not cfg['test_used_for_selection']
for n,h in cfg['code_sha256'].items():assert sha(ROOT/'code'/n)==h,n
from data import legacy
records=legacy.self_ids('val');assert len(records)==81
tests={r['camera'] for r in read(ROOT/'protocol/self_split.json')['records'] if r.get('subset')=='short' and r.get('split')=='test'}
assert len(tests)==82 and tests.isdisjoint(r['camera'] for r in records)
inputs=[]
for r in records:
 p=Path('@DATASET_ROOT@/processed_video')/r['camera']/'clip01.mp4';inputs.append(dict(camera=r['camera'],source=str(p),sha256=sha(p)))
sources={str(p):sha(p) for p in sorted((ROOT/'implementation').rglob('*.py')) if p.is_file()}
sources[str(legacy.__file__)]=sha(legacy.__file__);assert sources
checkpoints={}
for arm in cfg['arms']:
 rows=[]
 for step in cfg['candidate_steps']:
  p=Path(cfg['checkpoint_roots'][arm])/f'candidate_step{step:04d}.pth.tar';d=torch.load(p,map_location='cpu',weights_only=False)
  assert d['updates']==step and d['sampler_offset']==step*16
  assert all(torch.isfinite(t).all() for t in d['state_dict'].values())
  rows.append(dict(step=step,path=str(p),sha256=sha(p),bytes=p.stat().st_size));del d
 checkpoints[arm]=rows;print('CHECKPOINTS_AUDITED',arm,len(rows),flush=True)
atomic(ROOT/'protocol/input_checkpoint_audit.json',dict(passed=True,validation_videos=81,test_videos_disjoint=82,validation_inputs=inputs,checkpoints=checkpoints,original_i=dict(path=str(I),sha256=sha(I)),original_p=dict(path=str(P),sha256=sha(P)),sources=sources,no_source_or_checkpoint_modified=True,no_gpu_or_optimizer_update=True))
print('PREFLIGHT_PASSED',flush=True)
