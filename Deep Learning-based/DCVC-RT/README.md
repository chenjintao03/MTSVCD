# DCVC-RT for MTSVCD

This directory contains the codec and differentiable training source exported
from the MTSVCD experiment server. It derives from
[Microsoft DCVC-RT](https://github.com/microsoft/DCVC/tree/main/DCVC-family/DCVC-RT).
The differentiable training path contains local research changes.

- [Original project instructions](UPSTREAM_README.md)
- [MTSVCD reproduction guide](../../docs/REPRODUCIBILITY.md)
- [RT matched adaptation](../../experiments/rt_matched)
- [Four-dataset training](../../experiments/rt_endpoint)
- [Validation selection](../../experiments/rt_selection)
- [Original license](LICENSE.txt) and [third-party notice](NOTICE.txt)

Build the entropy-coder and CUDA extensions according to the original project
instructions before real-stream tests. Path/GPU templates in the experiment
sources are rendered by `tools/prepare_experiments.py` from the repository root.
