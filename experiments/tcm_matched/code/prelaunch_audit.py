"""CPU-only deployment checks; invokes the actual worker-generation code."""
from common import *
setup()
import ast,types,shutil
cfg=read(ROOT/'protocol/experiment.json')
assert cfg['frames']==5 and cfg['effective_batch']==4 and cfg['updates_per_quality_per_arm']==3000
assert all(sha(ROOT/path)==value for path,value in cfg['source_sha256'].items())
assert all(sha(ROOT/'protocol'/name)==value for name,value in cfg['input_protocol_sha256'].items())
assert read(ROOT/'diagnostic/input_gate.json')['status']=='passed'
assert read(ROOT/'diagnostic/cpu_model_gate.json')['status']=='passed'
for path in (ROOT/'code').glob('*.py'):ast.parse(path.read_text())
shared=ROOT/'cache/generated_worker_syntax';shared.mkdir(parents=True,exist_ok=True);atomic(shared/'manifest.json',dict(clips=[]));(shared/'inputs').mkdir(exist_ok=True)
tree=ast.parse((ROOT/'code/real_validation_gate.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare')
ns=dict(globals());exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_prepare_function','exec'),ns)
quality_reports=[]
for q in [1,2,3,4]:
    ns['a']=types.SimpleNamespace(quality=q);path=ns['prepare']('syntax_only',WEIGHTS/f'inter_psnr_TMM_q{q}.pth');source=(path/'code/learned_worker.py').read_text();ast.parse(source)
    quality_reports.append(dict(quality=q,generated_worker_sha256=sha(path/'code/learned_worker.py')))
atomic(ROOT/'protocol/generated_worker_syntax_audit.json',dict(passed=True,cpu_only=True,gpu_used=False,training_started=False,actual_prepare_function_used=True,qualities=quality_reports,source={p.name:sha(p) for p in (ROOT/'code').glob('*.py')}))
tree=ast.parse((ROOT/'code/evaluate.py').read_text());nodes=tree.body
start=next(i for i,n in enumerate(nodes) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='source')
end=next(i for i,n in enumerate(nodes[start+1:],start+1) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='n')
target=ROOT/'cache/formal_worker_syntax';(target/'code').mkdir(parents=True,exist_ok=True);selected={q:WEIGHTS/f'inter_psnr_TMM_q{q}.pth' for q in [1,2,3,4]}
ns=dict(globals());exec(compile(ast.Module(body=nodes[start:end],type_ignores=[]),'actual_evaluate_source_generation','exec'),ns)
assert ns['mapping']=={q-1:str(selected[q]) for q in selected}
for q in selected:assert ns['hashes'][q-1]==sha(selected[q])
roundtrip=dict(training_arm='surveillance',selected_p_sha256={str(k):v for k,v in ns['hashes'].items()});assert json.loads(json.dumps(roundtrip))==roundtrip
atomic(ROOT/'protocol/formal_worker_syntax_audit.json',dict(passed=True,cpu_only=True,gpu_used=False,training_started=False,actual_evaluate_generation_executed=True,qualities=4,original_p_mapping=ns['mapping'],paired_i_preserved=True,json_resume_key_roundtrip=True,generated_worker_sha256=sha(target/'code/learned_worker.py')))
print('TCM CPU prelaunch and actual four-quality validation/test worker generation passed; GPU gates still required',flush=True)
