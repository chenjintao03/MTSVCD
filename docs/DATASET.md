# Dataset and split conventions

## Continuous short videos

MTSVCD contains 543 short MP4 videos, generally two minutes at 30 fps and
640 × 384 pixels. Their recorded decoded lengths range from 3,506 to 3,601
frames, totaling 1,954,744 frames. Use the per-video counts in
`metadata/short_videos.csv` when locating temporal windows.

`splits/train.txt`, `splits/val.txt`, and `splits/test.txt` contain paths relative
to the extracted dataset root. The split is 380 / 81 / 82 videos. It is inherited
unchanged from the frozen version-2.0 manifest dated 2026-10-23. Its original
SHA-256 is recorded in `metadata/provenance.json`. Machine-specific paths and
original source titles have been omitted from the public manifest.

The 543 videos correspond to 533 related-viewpoint groups. Related videos stay
in the same split. A group is a reviewed association of viewpoints, rather than
a verified unique physical camera or geographic location. Category metadata
retains the original review status: `unreviewed` labels are initial labels,
while `accepted` labels were explicitly accepted during review.

The historical `split_order80` field in an earlier inventory is superseded by
this manifest. It must not be used to reconstruct the released split.

## Long-term sparse sequences

The 47 long sequences contain 24,609 sampled images, with approximately five
minutes between observations. They support observation of illumination,
weather, and scene changes over longer periods. They were not used in the
released compression training/validation/test split. `split` is therefore null
for long sequences in the manifest.

Short and long sequences share 28 related-viewpoint groups, giving
533 + 47 − 28 = 552 groups overall. Continuous frames and five-minute samples
serve different purposes and should not be concatenated into a single codec
training sequence.

## Compression evaluation windows

Compression evaluation uses three 96-frame windows for each of the 82 test
videos, totaling 246 windows. For a
video with N decoded frames, the zero-based start positions are:

1. Beginning: 0.
2. Middle: floor((N − 96) / 2).
3. End: N − 96.

Each window resets the codec state. I frames occur at positions 0, 32, and 64.
The original video is the aggregation unit; the three windows are not three
independent scenes. Input checksums and result manifests are generated locally.
The README presents the experiment results as paper figures.

## Local verification

```bash
python tools/validate_splits.py --dataset-root /path/to/extracted/MTSVCD
```

The validator checks video counts, list/manifest agreement, group isolation,
frame totals, and the presence of the extracted short videos and long folders.
