# Source and license notes

The DCVC-RT and DCVC-TCM implementations originate from
[Microsoft DCVC](https://github.com/microsoft/DCVC). This release was exported
from the MTSVCD experiment server on 2026-11-24. The server's matched-experiment
implementations take precedence over its upstream working copies where they
overlap. Per-file original and released SHA-256 values are recorded in
`metadata/server_source_provenance.json`.

Both codec directories retain Microsoft's MIT `LICENSE.txt` and its
`NOTICE.txt`, which includes notices for the CompressAI-derived code. Other
copyright and license notices embedded in source headers remain in place.
`UPSTREAM_README.md` preserves the server's upstream project instructions.
The RT differentiable training path contains local research changes and should
not be described as an official author release of training code.

DCVC-TCM's CMake build templates fetch pybind11 and ryg_rans from their upstream
repositories. DCVC-RT uses pybind11 and custom CUDA extensions. Consult
the retained build files and upstream instructions for their pinned versions
and license terms. `pytorch-msssim` is installed as a dependency.

The directory organization follows the example of
[SCVCD-5K](https://github.com/yuanhui0325/SCVCD-5K). No SCVCD-5K dataset videos
are redistributed here. The dataset download and codec-source licenses are
separate; this repository does not assign a new blanket license to the
underlying surveillance footage.
