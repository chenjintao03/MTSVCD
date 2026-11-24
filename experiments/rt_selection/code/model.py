"""RT adaptation using the paper's hierarchical distortion objective.

The differentiable RT path is an explicitly disclosed adaptation; the released
RT inference code does not supply the authors' complete training schedule.
"""
import torch
from src.models.training_model import VideoForTraining, IntraForTraining, yuv420_metrics
from src.utils.transforms import ycbcr2rgb

class Video(VideoForTraining):
    distortion_weights=(0.5,1.2,0.5,0.9)
    def forward(self,frames,base_qp,intra_model,lambdas,detach_refs=False,noisy_rate=None):
        if frames.shape[1]>32:
            values=[self.forward(frames[:,n:n+32],base_qp,intra_model,lambdas,detach_refs,noisy_rate) for n in range(0,frames.shape[1],32)]
            return {k:torch.stack([r[k] for r in values]).mean() for k in values[0]}
        assert frames.ndim==5 and 2<=frames.shape[1]<=32
        if noisy_rate is None:noisy_rate=self.training
        self.clear_dpb();self.set_curr_poc(0)
        with torch.no_grad():first=intra_model(frames[:,0],base_qp)
        self.add_ref_frame(frame=first)
        losses=[];rates=[];distortions=[];psnrs=[]
        for n in range(1,frames.shape[1]):
            qp=base_qp+self.qp_shift[self.index_map[n%8]]
            r=self.forward_one_frame(frames[:,n],qp,noisy_rate)
            rgb=(ycbcr2rgb(frames[:,n])-ycbcr2rgb(r['x_hat'])).square().mean((1,2,3))
            d=.8*r['distortion']+.2*rgb
            losses.append(r['bpp']+lambdas*self.distortion_weights[(n-1)%4]*d)
            rates.append(r['bpp']);distortions.append(d);psnrs.append(r['psnr'])
            if detach_refs:self.detach_reference()
        self.clear_dpb()
        return dict(loss=torch.stack(losses).mean(),p_bpp_estimated=torch.stack(rates).mean(),p_distortion=torch.stack(distortions).mean(),p_psnr_yuv=torch.stack(psnrs).mean())
