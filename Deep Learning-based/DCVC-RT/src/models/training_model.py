# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Differentiable fine-tuning paths for the released DCVC-RT architecture.

This is an adaptation of the STE/noisy-rate training approach in the repository's
root training code, not the authors' unreleased DCVC-RT training recipe. No new
parameters are added, so exported state dictionaries load into the original DMC.
Import this module before constructing any model used for training. Its explicit
Torch routing is process-wide: run actual bitstream tests in a separate process.
"""

import math

import torch
import torch.nn.functional as F

from ..layers import cuda_inference as _cuda_inference

# These extensions cache/fuse weights and do not implement training backward.
_cuda_inference.CUSTOMIZED_CUDA_INFERENCE = False
from ..layers import layers as _layers  # noqa: E402

_layers.CUSTOMIZED_CUDA_INFERENCE = False
from . import image_model as _image_model  # noqa: E402
from . import video_model as _video_model  # noqa: E402

_image_model.CUSTOMIZED_CUDA_INFERENCE = False
_video_model.CUSTOMIZED_CUDA_INFERENCE = False


def disable_cuda_inference():
    """Disable every copied inference flag before creating fresh model objects."""
    for module in (_cuda_inference, _layers, _image_model, _video_model):
        module.CUSTOMIZED_CUDA_INFERENCE = False


class LowerBound(torch.autograd.Function):
    """The root codec's lower bound, which permits gradients toward the bound."""

    @staticmethod
    def forward(ctx, values, bound):
        ctx.save_for_backward(values)
        ctx.bound = bound
        return values.clamp_min(bound)

    @staticmethod
    def backward(ctx, gradient):
        (values,) = ctx.saved_tensors
        return gradient * ((values >= ctx.bound) | (gradient < 0)), None


def quantize_ste(values):
    rounded = values.round().clamp(-128, 127)
    return values + (rounded - values).detach()


def _select(table, qp):
    return torch.index_select(table, 0, qp.long())


def _masked_quantize(y, scales, means, mask, force_zero_thres):
    residual = (y - means) * mask
    symbols = quantize_ste(residual)
    active = mask
    if force_zero_thres is not None:
        active = mask * (scales > force_zero_thres).to(mask.dtype)
        symbols = symbols * active
    reconstructed = symbols + means * mask
    return residual, symbols, reconstructed, active


def gaussian_bits(residual, scales, active, noisy):
    """Continuous Gaussian CDF rate estimate; this is not an actual bit count."""
    values = residual
    if noisy:
        values = values + torch.empty_like(values).uniform_(-0.5, 0.5)
    else:
        values = values.round().clamp(-128, 127)
    values = values.float().abs()
    # Match the range supported by RT's arithmetic coding scale table.
    sigma = LowerBound.apply(scales.float(), 0.11).clamp_max(16.0)
    root_two = math.sqrt(2.0)
    upper = torch.erfc((values - 0.5) / (sigma * root_two))
    lower = torch.erfc((values + 0.5) / (sigma * root_two))
    probability = (0.5 * (upper - lower)).clamp_min(1e-9)
    return (-torch.log2(probability) * active).sum(dim=(1, 2, 3))


def learned_z_bits(bit_estimator, z, qp, noisy):
    values = z + torch.empty_like(z).uniform_(-0.5, 0.5) if noisy else z.round().clamp(-128, 127)
    # Keep small CDF differences in float32 even if a caller later enables AMP.
    values = values.float()
    upper = bit_estimator(values + 0.5, qp.long())
    lower = bit_estimator(values - 0.5, qp.long())
    probability = (upper - lower).clamp(min=1e-9, max=1.0)
    return -torch.log2(probability).sum(dim=(1, 2, 3))


def yuv420_metrics(x, reconstructed):
    """Normalized 6:1:1 MSE and the test program's weighted YUV PSNR."""
    difference_y = (x[:, :1] - reconstructed[:, :1]).square().mean(dim=(1, 2, 3))
    source_uv = F.avg_pool2d(x[:, 1:], 2, 2)
    reconstructed_uv = F.avg_pool2d(reconstructed[:, 1:], 2, 2)
    difference_uv = (source_uv - reconstructed_uv).square().mean(dim=(2, 3))
    component_mse = torch.stack((difference_y, difference_uv[:, 0], difference_uv[:, 1]), 1)
    weights = component_mse.new_tensor((6.0, 1.0, 1.0))
    distortion = (component_mse * weights).sum(1) / 8.0
    component_psnr = -10.0 * torch.log10(component_mse.clamp_min(1e-12))
    psnr = (component_psnr * weights).sum(1) / 8.0
    return distortion, psnr, component_psnr


class IntraForTraining(_image_model.DMCI):
    """Frozen I-frame reconstruction without C++ entropy coding or CUDA proxies."""

    def __init__(self, *args, force_zero_thres=0.12, **kwargs):
        disable_cuda_inference()
        super().__init__(*args, **kwargs)
        self.force_zero_thres = force_zero_thres

    def forward(self, x, qp):
        q_encoder = _select(self.q_scale_enc, qp)
        q_decoder = _select(self.q_scale_dec, qp)
        y = self.enc(x, q_encoder)
        z = self.hyper_enc(self.pad_for_y(y))
        z_hat = quantize_ste(z)
        parameters = self.y_prior_fusion(self.hyper_dec(z_hat))
        parameters = parameters[:, :, :y.shape[-2], :y.shape[-1]].contiguous()
        q_enc, q_dec, scales, means = self.separate_prior(parameters, is_video=False)
        common = self.y_spatial_prior_reduction(parameters)
        normalized = y * q_enc
        masks = self.get_mask_4x(*y.shape, y.dtype, y.device)
        accumulated = torch.zeros_like(y)
        adaptors = (self.y_spatial_prior_adaptor_1,
                    self.y_spatial_prior_adaptor_2,
                    self.y_spatial_prior_adaptor_3)
        for part, mask in enumerate(masks):
            if part:
                prior_input = torch.cat((accumulated, common), dim=1)
                scales, means = self.y_spatial_prior(adaptors[part - 1](prior_input)).chunk(2, 1)
            _, _, reconstructed, _ = _masked_quantize(
                normalized, scales, means, mask, self.force_zero_thres)
            accumulated = accumulated + reconstructed
        return self.dec(accumulated * q_dec, q_decoder).clamp(0.0, 1.0)


class VideoForTraining(_video_model.DMC):
    """Trainable RT with the original module names and tensor shapes."""

    index_map = (0, 1, 0, 2, 0, 2, 0, 2)

    def __init__(self, force_zero_thres=0.12):
        disable_cuda_inference()
        super().__init__()
        self.force_zero_thres = force_zero_thres

    def forward_one_frame(self, x, qp, noisy_rate=True):
        q_encoder = _select(self.q_encoder, qp)
        q_decoder = _select(self.q_decoder, qp)
        q_feature = _select(self.q_feature, qp)
        q_recon = _select(self.q_recon, qp)
        feature = self.apply_feature_adaptor()
        context, temporal_context = self.feature_extractor(feature, q_feature)
        y = self.encoder(x, context, q_encoder)
        z = self.hyper_encoder(self.pad_for_y(y))
        parameters = self.res_prior_param_decoder(quantize_ste(z), temporal_context)
        q_dec, scales, means = parameters.chunk(3, 1)
        q_dec = LowerBound.apply(q_dec, 0.5)
        normalized = y / q_dec
        masks = self.get_mask_2x(*y.shape, y.dtype, y.device)
        residual0, _, reconstructed0, active0 = _masked_quantize(
            normalized, scales, means, masks[0], self.force_zero_thres)
        bits_y = gaussian_bits(residual0, scales, active0, noisy_rate)
        scales, means = self.y_spatial_prior(torch.cat((reconstructed0, parameters), 1)).chunk(2, 1)
        residual1, _, reconstructed1, active1 = _masked_quantize(
            normalized, scales, means, masks[1], self.force_zero_thres)
        bits_y = bits_y + gaussian_bits(residual1, scales, active1, noisy_rate)
        y_hat = (reconstructed0 + reconstructed1) * q_dec
        x_hat, feature = self.get_recon_and_feature(y_hat, context, q_decoder, q_recon)
        self.add_ref_frame(feature=feature, frame=x_hat)
        bits_z = learned_z_bits(self.bit_estimator_z, z, qp, noisy_rate)
        distortion, psnr, component_psnr = yuv420_metrics(x, x_hat)
        return {"x_hat": x_hat, "bits_y": bits_y, "bits_z": bits_z,
                "bpp": (bits_y + bits_z) / (x.shape[-2] * x.shape[-1]),
                "distortion": distortion, "psnr": psnr, "component_psnr": component_psnr}

    def detach_reference(self):
        for reference in self.dpb:
            if reference.frame is not None:
                reference.frame = reference.frame.detach()
            if reference.feature is not None:
                reference.feature = reference.feature.detach()

    def forward(self, frames, base_qp, intra_model, lambdas, detach_refs=True,
                noisy_rate=None):
        """Return P-frame RD averaged over a clip; I-frame cost is excluded.

        frames: [B,T,3,H,W] YCbCr in [0,1], base_qp/lambdas: [B]. Frozen
        I reconstruction uses the matching base QP. Reference detachment enables
        truncated temporal gradients, while False optimizes the complete clip.
        """
        if frames.ndim != 5 or frames.shape[1] < 2:
            raise ValueError("Expected [B,T,3,H,W] with at least two frames")
        if frames.shape[2] != 3 or any(size % 64 for size in frames.shape[-2:]):
            raise ValueError("Expected three channels and dimensions divisible by 64")
        if base_qp.shape != (frames.shape[0],) or lambdas.shape != base_qp.shape:
            raise ValueError("base_qp and lambdas must each have one value per clip")
        if bool(((base_qp < 0) | (base_qp >= self.get_qp_num())).any()):
            raise ValueError("Base QP must be in [0,63]")
        if noisy_rate is None:
            noisy_rate = self.training
        self.clear_dpb()
        self.set_curr_poc(0)
        with torch.no_grad():
            first_frame = intra_model(frames[:, 0], base_qp)
        self.add_ref_frame(frame=first_frame)
        results = []
        losses = []
        for frame_index in range(1, frames.shape[1]):
            bank_index = self.index_map[frame_index % 8]
            qp = base_qp + self.qp_shift[bank_index]
            result = self.forward_one_frame(frames[:, frame_index], qp, noisy_rate)
            losses.append(lambdas * result["distortion"] + result["bpp"])
            results.append(result)
            if detach_refs:
                self.detach_reference()
        # The returned losses retain all required autograd edges.
        self.clear_dpb()
        return {"loss": torch.stack(losses).mean(),
                "p_bpp_estimated": torch.stack([r["bpp"] for r in results]).mean(),
                "p_distortion": torch.stack([r["distortion"] for r in results]).mean(),
                "p_psnr_yuv": torch.stack([r["psnr"] for r in results]).mean(),
                "p_psnr_y": torch.stack([r["component_psnr"][:, 0] for r in results]).mean(),
                "p_psnr_u": torch.stack([r["component_psnr"][:, 1] for r in results]).mean(),
                "p_psnr_v": torch.stack([r["component_psnr"][:, 2] for r in results]).mean()}
