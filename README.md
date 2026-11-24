# MTSVCD

**Multi-Timescale Surveillance Video Compression Dataset**

Project repository for *A Multi-Timescale Surveillance Video Dataset for Compression*.
MTSVCD combines continuous surveillance videos for compression experiments with
long-term, sparsely sampled sequences for observing changes in fixed-view scenes.

## Dataset download

[Download MTSVCD from Baidu Netdisk](https://pan.baidu.com/s/1Sp_txPKckMhYkBKZ4CjNBg)

**Extraction code / 提取码: `gj2k`**

The video and image data are distributed through the download link. This GitHub
repository contains the split lists, metadata, codec source projects, and experiment configurations.

| Subset | Sequences | Sampling | Use |
|---|---:|---|---|
| Short-term | 543 continuous videos; 1,954,744 frames | Approximately 2 minutes per video, 30 fps, 640 × 384 | Compression training, validation, and testing |
| Long-term | 47 sparse sequences; 24,609 images | Approximately one frame every 5 minutes | Long-term observation and descriptive analysis |

The short videos form 533 related-viewpoint groups. The 47 long sequences form
47 groups; 28 groups are shared between the subsets, giving 552 groups overall.
Sequence counts and related-viewpoint group counts are different quantities.

## Fixed train / validation / test split

| Partition | Short videos | List |
|---|---:|---|
| Training | 380 | [train.txt](splits/train.txt) |
| Validation | 81 | [val.txt](splits/val.txt) |
| Test | 82 | [test.txt](splits/test.txt) |

The assignments are preserved from the frozen split manifest, version 2.0
(2026-10-23). No short-term related-viewpoint group crosses partitions. The
long-term subset has no training, validation, or test assignment.

- [Portable split manifest](metadata/split_manifest.json)
- [Short-video metadata](metadata/short_videos.csv) and [long-sequence metadata](metadata/long_sequences.csv)
- [Dataset structure and counting conventions](docs/DATASET.md)
- [Frozen compression-test windows](metadata/evaluation_windows.csv)

Each list contains one video path per line, relative to the extracted dataset root:

```text
<dataset-root>/
├── processed_video/
│   ├── cam0001/clip01.mp4
│   └── ...
└── Long_video/
    ├── cam10/
    └── ...
```

Validate the release and, optionally, the extracted data:

```bash
python tools/validate_splits.py
python tools/validate_splits.py --dataset-root /path/to/MTSVCD
```

## Codec projects and experiments

Codec implementations are under `Deep Learning-based/`:

- [DCVC-RT](Deep%20Learning-based/DCVC-RT): the server's codec implementation and differentiable training path.
- [DCVC-TCM](Deep%20Learning-based/DCVC-TCM): the server's codec implementation, with its fine-tuning adapter in `experiments/tcm_matched/`.
- [Experiment sources and configurations](experiments): RT/TCM adaptation, four-dataset RT comparisons, validation selection, and native-YUV420 bitstream evaluation.
- [Reproduction instructions](docs/REPRODUCIBILITY.md): environments, pretrained weights, paths, preparation, and execution order.
- [Actual reported checkpoint selection](metadata/reported_checkpoint_selection.csv).
- [Server-source provenance and SHA-256 identities](metadata/server_source_provenance.json).

The codec projects derive from [Microsoft DCVC](https://github.com/microsoft/DCVC).
Their original licenses and notices are retained in each project. Experimental
scripts use path and GPU-UUID templates, rendered into a separate workspace by
`tools/prepare_experiments.py`. Datasets, pretrained/fine-tuned weights, and
generated bitstreams are stored outside Git.

## Acknowledgments and licenses

We acknowledge the authors of DCVC-RT and DCVC-TCM.
See the [license and source notes](docs/THIRD_PARTY.md) and the original
`LICENSE.txt` / `NOTICE.txt` files in each codec directory.
