# Server experiment workflows

These directories contain the server experiment sources and configuration
files. They use `@WORKSPACE@`, `@DATASET_ROOT@`, and related templates.
Render them with `tools/prepare_experiments.py` before execution; do not run
the template files directly.

See [the reproduction guide](../docs/REPRODUCIBILITY.md) for dependencies,
workspace setup, and the workflow order. Input manifests, per-video results,
and runtime records are created locally. The public experiment presentation
consists of the paper figures in the [project README](../README.md#experimental-results).
