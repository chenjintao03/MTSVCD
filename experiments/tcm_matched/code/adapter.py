"""Differentiable TCM adapter extracted unchanged from the previously audited adapter.
No RT quality bank or hierarchical weighting is applied to this TCM model.
The official inference architecture is preserved in the isolated implementation.
"""
import torch
from common import LAMBDAS,state,WEIGHTS,sha
from src.models.video_net_dmc import DMC
from src.models.priors import IntraNoAR

def rd_forward(m, x, ref, feature=None, training=False, noise=True):
    # In evaluation, each operation and its order match forward_one_frame.
    def quant(v):
        return v + (v.round() - v).detach() if training else v.round()
    def rate_value(v):
        return v + torch.empty_like(v).uniform_(-.5, .5) if training and noise else quant(v)
    mv_y = m.mv_encoder(m.optic_flow(x, ref))
    mv_z = m.mv_prior_encoder(mv_y)
    mv_params = m.mv_prior_decoder(quant(mv_z))
    mv_scales, mv_means = mv_params.chunk(2, 1)
    mv_res = mv_y - mv_means
    mv_q = quant(mv_res)
    mv = m.mv_decoder(mv_q + mv_means)
    c1, c2, c3, _ = m.motion_compensation(ref, feature, mv)
    y = m.contextual_encoder(x, c1, c2, c3)
    z = m.contextual_hyper_prior_encoder(y)
    hp = m.contextual_hyper_prior_decoder(quant(z))
    tp = m.temporal_prior_encoder(c1, c2, c3)
    scales, means = m.contextual_entropy_parameter(torch.cat((tp, hp), dim=1)).chunk(2, 1)
    y_res = y - means
    y_q = quant(y_res)
    f, recon = m.recon_generation_net(m.contextual_decoder(y_q + means, c2, c3), c1)
    # Retain the original Laplace family, scale bounds and probability floor.
    by, _ = m.get_y_bits_probs(rate_value(y_res), scales)
    bmv, _ = m.get_y_bits_probs(rate_value(mv_res), mv_scales)
    bz, _ = m.get_z_bits_probs(rate_value(z), m.bit_estimator_z)
    bmz, _ = m.get_z_bits_probs(rate_value(mv_z), m.bit_estimator_z_mv)
    n = x.shape[0] * x.shape[2] * x.shape[3]
    bpp = by/n + bz/n + bmv/n + bmz/n
    mse = torch.mean((recon - x).square())
    return {'bpp': bpp, 'mse_loss': mse, 'recon_image': recon, 'feature': f,
            'bpp_y': by/n, 'bpp_z': bz/n, 'bpp_mv_y': bmv/n, 'bpp_mv_z': bmz/n}

def models(q,checkpoint=None,device='cuda'):
    ip=WEIGHTS/f'intra_psnr_TMM_q{q}.pth.tar';pp=WEIGHTS/f'inter_psnr_TMM_q{q}.pth'
    intra=IntraNoAR.from_state_dict(state(ip)).to(device).eval();intra.requires_grad_(False)
    inter=DMC().to(device);inter.load_dict(state(checkpoint or pp),strict=True)
    return intra,inter,dict(i_path=str(ip),i_sha256=sha(ip),p_original_path=str(pp),p_original_sha256=sha(pp))
def sequence_loss(inter,intra,frames,q,training=True,noise=True):
    assert frames.ndim==5 and frames.shape[1]==5
    with torch.no_grad():ref=intra(frames[:,0])['x_hat'].clamp(0,1)
    feature=None;losses=[];rates=[];errors=[]
    for n in range(1,5):
        out=rd_forward(inter,frames[:,n],ref,feature,training,noise)
        losses.append(out['bpp']+LAMBDAS[q]*out['mse_loss']);rates.append(out['bpp']);errors.append(out['mse_loss'])
        ref=out['recon_image'].clamp(0,1);feature=out['feature']
    return dict(loss=torch.stack(losses).mean(),p_bpp_estimated=torch.stack(rates).mean(),p_rgb_mse=torch.stack(errors).mean())
