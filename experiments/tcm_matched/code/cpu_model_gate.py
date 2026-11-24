from common import *
setup();os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,traceback
from adapter import models
torch.set_num_threads(2)
report=dict(status='running',gpu_used=False,checks=[],kind='CPU strict checkpoint loading only; no forward, gradients, coding or training')
def save():atomic(ROOT/'diagnostic/cpu_model_gate.json',report)
save()
try:
    assert not torch.cuda.is_available()
    for q in [1,2,3,4]:
        i,p,meta=models(q,device='cpu')
        assert all(x.device.type=='cpu' for n in [i,p] for x in n.parameters())
        assert all(not x.requires_grad for x in i.parameters())
        assert all(torch.isfinite(x).all() for n in [i,p] for x in n.state_dict().values())
        report['checks'].append(dict(quality=q,strict_original_load=True,intra_parameters=sum(x.numel() for x in i.parameters()),p_parameters=sum(x.numel() for x in p.parameters()),all_cpu=True,all_finite=True,**meta));save();del i,p
    report.update(status='passed',gpu_gradient_and_real_stream_gates_still_required=True);save()
except BaseException:report.update(status='failed',error=traceback.format_exc());save();raise
