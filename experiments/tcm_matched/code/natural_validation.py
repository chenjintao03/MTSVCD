"""Report natural validation only after final actual-stream checkpoint selection."""
from common import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['surveillance','vimeo'],required=True);p.add_argument('--quality',type=int,choices=[1,2,3,4],required=True);p.add_argument('--gpu',type=int,choices=[4,5],required=True);a=p.parse_args();setup(a.gpu)
from adapter import models
from validation import cached_validation
from data import Validation
folder=ROOT/a.arm/f'q{a.quality}';selection=read(folder/'selection.json');assert read(folder/'real_validation_gate.json')['passed']
checkpoint=folder/'selected.pth.tar';intra,inter,meta=models(a.quality,checkpoint)
# Original models use their original file identity in the shared cache after exact tensor equality.
original=WEIGHTS/f'inter_psnr_TMM_q{a.quality}.pth'
if not selection['accepted']:
    d1=state(checkpoint);d2=state(original);assert d1.keys()==d2.keys() and all(d1[k].equal(d2[k]) for k in d1);identity=original;path=ROOT/f'original_validation/q{a.quality}/natural.json'
else:identity=checkpoint;path=folder/'natural_cache.json'
def progress(n,total):atomic(folder/'natural_status.json',dict(status='running',completed=n,total=total,final_selected_step=selection['step']))
value=cached_validation(inter,intra,a.quality,Validation('vimeo'),progress,path,identity,not selection['accepted'])
atomic(folder/'natural_validation.json',dict(final_selected_checkpoint_sha256=sha(checkpoint),model_identity_checkpoint_sha256=sha(identity),selection=selection,metrics=value,test_used_for_selection=False))
atomic(folder/'natural_status.json',dict(status='complete',completed=7824,total=7824,final_selected_step=selection['step']))
