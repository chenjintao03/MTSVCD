# Reproducing the MTSVCD codec experiments

## What is included

The codec projects and experiment scripts were exported from the experiment
server on 2026-11-24. `metadata/server_source_provenance.json` records the
original and released file hashes. Only machine-specific paths and GPU UUIDs
were converted to templates. Training objectives and codec computations are
preserved.

| Directory | Purpose |
|---|---|
| `Deep Learning-based/DCVC-RT` | RT codec, C++/CUDA extension sources, and local differentiable training implementation |
| `Deep Learning-based/DCVC-TCM` | TCM codec and CMake entropy-coder build sources |
| `experiments/rt_matched` | Paired MTSVCD/Vimeo RT adaptation with validation and real-stream checks |
| `experiments/tcm_matched` | Four-quality TCM adaptation, gradient gates, validation, and real-stream checks |
| `experiments/rt_endpoint` | Four-dataset RT adaptation to 3000 updates |
| `experiments/rt_selection` | Saved-checkpoint validation and selected-model testing |
| `experiments/rt_vimeo600` | Final Vimeo-600 checkpoint bitstream test |
| `experiments/native420_benchmark` | Shared native-YUV420 baseline worker, metrics, and input identities |
| `experiments/cross_dataset` | Cross-dataset evaluation sources and configuration |

These are staged experiment workflows. The controllers depend on preceding
input/model/gradient gates and on generated runtime manifests. Preparation
renders paths and source identities; the real gates must still be run. No old
successful GPU-gate result is used to certify a new workspace.

## Environments and model files

The captured package versions are in `metadata/runtime_versions.json`:

| | RT environment | TCM environment |
|---|---|---|
| Python | 3.12.16 | 3.11.15 |
| PyTorch | 2.6.0+cu126 | 2.6.0+cu126 |
| NumPy | 1.26.4 | 2.1.2 |
| SciPy | 1.15.3 | 1.15.2 |
| Pillow | 11.1.0 | 11.0.0 |
| pybind11 | 2.13.6 | 3.0.4 |

The original `requirements.txt` and `UPSTREAM_README.md` are retained in each
codec project. Their historical minimum/example environments differ from the
captured experiment environments. Training and bitstream workflows use Linux
or WSL, FFmpeg/ffprobe, a compatible CUDA toolchain, and separately built codec
extensions. For RT, follow the two `pip install .` extension steps in its
upstream README. For TCM, follow the `src/cpp` CMake build and make the built
`MLCodec_*` modules importable in the TCM environment.

Obtain the original model files from the download instructions in the retained
upstream READMEs:

- RT: `cvpr2025_image.pth.tar` and `cvpr2025_video.pth.tar`.
- TCM: `intra_psnr_TMM_q1.pth.tar` … `intra_psnr_TMM_q4.pth.tar` and
  `inter_psnr_TMM_q1.pth` … `inter_psnr_TMM_q4.pth`.

Store the models outside the checkout and set `RT_WEIGHTS` / `TCM_WEIGHTS` in
the path configuration. Fine-tuned weights and generated bitstreams are not
part of this Git source release.

## Prepare a workspace

1. Download and extract MTSVCD. Validate the fixed split with
   `python tools/validate_splits.py --dataset-root /path/to/dataset`.
2. Copy `configs/paths.example.json` to a local configuration outside Git and
   fill in its paths. `DATASET_ROOT` contains `processed_video/` and
   `Long_video/`. Supply the Vimeo septuplet root when running the Vimeo arm;
   the UA-DETRAC and ShanghaiTech arms require their separately obtained data
   and training manifests.
3. Set `GPU4_UUID` and `GPU5_UUID` to the UUIDs of physical GPUs 4 and 5, obtained
   with `nvidia-smi --query-gpu=index,uuid --format=csv,noheader`.
4. Use a new, empty workspace outside Git. On the original Windows/WSL setup,
   keep workspaces, caches, test inputs, logs, reconstructions, bitstreams, and
   checkpoints on D: / `/mnt/d/dataset`.
5. Render the templates:

```bash
python tools/prepare_experiments.py --config /path/to/local_paths.json
```

Preparation copies the source into the workspace, generates portable split
compatibility files, and records new fingerprints for the rendered files. It
does not run models or create successful experiment-gate records. Retain both
the release provenance and the workspace configuration with a reproduction.

## Inputs, execution order, and checks

`metadata/evaluation_windows.csv` lists the 246 test windows and their exact
raw input identities. To prepare their native-YUV420 data in the configured
workspace:

```bash
python tools/prepare_test_inputs.py --config /path/to/local_paths.json
```

The tool checks the source MP4 and decoded input hashes. It does not resize,
remap range, or round-trip the inputs through RGB. Run the shared baseline
worker with `--pilot` before a full test. The worker checks real bitstreams
and, in its pilot, independent decoding.

For the RT matched workflow, run `preflight.py` and `gradient_gate.py` in its
rendered `code/` directory before `train_matched.py`. `controller.py` encodes
the complete sequence of preflight, gradient gate, paired training, validation
guard, and testing. It also requires the shared benchmark inputs and baseline
results for comparisons.

For TCM, first run `input_gate.py`, `cpu_model_gate.py`, and
`prelaunch_audit.py` in the TCM environment. Its controller waits for RT
completion, executes the four quality-specific GPU gates, and then runs the
training/validation/test stages. Do not replace required gate files with
manually written pass markers. Cross-dataset and later four-arm controllers
retain their recorded dependencies on the preceding runs and source manifests.

The repository was checked for split integrity, source completeness, and
Python syntax. Publishing this snapshot did not rerun training or GPU
bitstream tests; a new environment must perform its own build and gates.

## Reported protocol and checkpoint scope

RT uses seven-frame training sequences, a 256 × 256 crop, a frozen I codec,
and effective batch size 16. TCM uses five-frame sequences, a frozen I codec,
and four accumulated microbatches of one sample. Detailed objectives,
learning rates, seeds, and validation rules are in each `protocol/experiment.json`.

The final RT comparison uses the recorded checkpoints: Vimeo-90K 600 updates,
UA-DETRAC 500, ShanghaiTech 3000, and MTSVCD 2500. These are distinct from the
uniform-3000 endpoint experiment and the earlier matched selection rule.
MTSVCD TCM uses q1 at 500 updates, the original q2 model, and q3/q4 at 3000.
See `metadata/reported_checkpoint_selection.csv` and
`metadata/reproducibility.json`. The released historical selection scripts
should not be assumed to automatically select every final reported checkpoint.

The frozen native-YUV420 test has 82 original videos, three 96-frame windows
per video, four operating points, and I refresh every 32 frames. RT uses QPs
0, 21, 42, and 63, with reset interval 64. TCM uses its four paired quality
models. Rate is computed from actual stream bytes, including headers and I/P
frames. The shared native-YUV420 adapter uses (6Y + U + V) / 8 for the component
metric aggregation. Keep the result manifest's valid-video set and aggregation
rule when reproducing a reported RD curve or BD-rate value.

The long-term five-minute samples were used for observation and descriptive
analysis. They are not part of these continuous-video codec training or test
inputs.
