# Codec setup

The codec projects are under `Deep Learning-based/DCVC-RT` and
`Deep Learning-based/DCVC-TCM`. Each retains its upstream README, source,
license, and third-party notice. The RT project also includes the server's
local differentiable training path.

## Environments and builds

`metadata/runtime_versions.json` records the captured environment versions:
RT used Python 3.12.16 with PyTorch 2.6.0+cu126, and TCM used Python 3.11.15
with PyTorch 2.6.0+cu126. The upstream READMEs and requirements describe their
original example environments and build instructions.

Build RT's entropy-coder and CUDA extensions, and TCM's CMake entropy-coder
extensions, in the corresponding environments. Linux/WSL workflows require
FFmpeg and a compatible CUDA toolchain.

## Model files and data

Obtain the original pretrained models from the download instructions in each
codec's `UPSTREAM_README.md`. RT uses `cvpr2025_image.pth.tar` and
`cvpr2025_video.pth.tar`. TCM uses four paired
`intra_psnr_TMM_q*.pth.tar` and `inter_psnr_TMM_q*.pth` models.

The dataset download link and fixed split lists are in the [project README](../README.md).
Validate the extracted dataset with:

```bash
python tools/validate_splits.py --dataset-root /path/to/MTSVCD
```

Use the codec entry points and input formats described by each upstream README.
Store generated reconstructions, bitstreams, metrics, logs, and checkpoints in
a separate workspace. On the original Windows/WSL setup, outputs belong on
D: / `/mnt/d/dataset`.

The [experiments directory](../experiments) contains only paper figures.
