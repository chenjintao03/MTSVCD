# DCVC-TCM for MTSVCD

This directory contains the codec source exported from the MTSVCD experiment
server. It derives from
[Microsoft DCVC-TCM](https://github.com/microsoft/DCVC/tree/main/DCVC-family/DCVC-TCM).

- [Original project instructions](UPSTREAM_README.md)
- [MTSVCD reproduction guide](../../docs/REPRODUCIBILITY.md)
- [Fine-tuning adapter and workflow](../../experiments/tcm_matched)
- [Original license](LICENSE.txt) and [third-party notice](NOTICE.txt)

Build the CMake entropy-coder extensions according to the original project
instructions. The fine-tuning adapter preserves the codec architecture and
adds differentiable training operations in `experiments/tcm_matched/code/adapter.py`.
