# Server experiment workflows

These directories contain the actual server experiment sources and selected
protocol files. They use `@WORKSPACE@`, `@DATASET_ROOT@`, and related templates.
Render them with `tools/prepare_experiments.py` before execution; do not run
the template files directly.

See [the reproduction guide](../docs/REPRODUCIBILITY.md) for dependencies,
workflow order, fresh input/model/gradient gates, and the exact final reported
checkpoint selections. Checkpoints, datasets, original host settings, login
information, and old successful runtime gate markers are not included.
