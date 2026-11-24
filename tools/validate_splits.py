"""Validate the released split and optionally check extracted dataset files."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


def validate(repo, dataset_root=None):
    manifest = json.loads((repo / 'metadata/split_manifest.json').read_text(encoding='utf-8'))
    short = [r for r in manifest['records'] if r['subset'] == 'short']
    long = [r for r in manifest['records'] if r['subset'] == 'long']
    assert len(short) == 543 and len(long) == 47
    assert len({r['id'] for r in manifest['records']}) == 590
    assert len({r['path'] for r in short}) == 543
    assert Counter(r['split'] for r in short) == {'train': 380, 'val': 81, 'test': 82}
    assert all(r['split'] is None for r in long)
    groups = defaultdict(set)
    for row in short:
        p = PurePosixPath(row['path'])
        assert not p.is_absolute() and '..' not in p.parts and '\\' not in row['path']
        groups[row['reviewed_group_id']].add(row['split'])
    assert len(groups) == 533
    assert all(len(s) == 1 for s in groups.values()), 'A related group crosses splits'
    for split in ('train', 'val', 'test'):
        actual = (repo / f'splits/{split}.txt').read_text(encoding='utf-8').splitlines()
        expected = sorted(r['path'] for r in short if r['split'] == split)
        assert actual == expected, f'{split}.txt disagrees with the manifest'
    assert sum(r['frames'] for r in short) == 1954744
    if dataset_root is not None:
        missing = [r['path'] for r in short if not (dataset_root / r['path']).is_file()]
        assert not missing, f'{len(missing)} missing videos, e.g. {missing[:3]}'
        missing_long = [r['path'] for r in long if not (dataset_root / r['path']).is_dir()]
        assert not missing_long, f'{len(missing_long)} missing long-sequence directories'
    print('PASS: 543 short videos = 380 train + 81 val + 82 test; 47 unpartitioned long sequences.')
    print('PASS: 533 short related-viewpoint groups; no group crosses train/val/test.')
    if dataset_root is not None:
        print('PASS: all video files and long-sequence directories exist under the dataset root.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    validate(args.repo_root, args.dataset_root)
