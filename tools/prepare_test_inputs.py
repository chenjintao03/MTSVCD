"""Extract the frozen test windows as native 8-bit YUV420 and verify their hashes."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def prepare(config, ffmpeg, limit):
    settings = json.loads(config.read_text(encoding='utf-8'))
    workspace = Path(settings['WORKSPACE']).resolve()
    dataset = Path(settings['DATASET_ROOT']).resolve()
    job = workspace / 'native420_benchmark'
    output = job / 'inputs/yuv420'
    output.mkdir(parents=True, exist_ok=True)
    temporary = workspace / 'tmp'
    temporary.mkdir(exist_ok=True)
    for key in ('TEMP', 'TMP', 'TMPDIR'):
        os.environ[key] = str(temporary)
    ffmpeg = shutil.which(ffmpeg) or ffmpeg
    rows = list(csv.DictReader((REPO / 'metadata/evaluation_windows.csv').open(encoding='utf-8', newline='')))
    selected = rows[:limit] if limit else rows
    verified_sources = set()
    verified = []
    for row in selected:
        source = dataset / 'processed_video' / row['camera'] / 'clip01.mp4'
        if source not in verified_sources:
            if sha(source) != row['source_sha256']:
                raise ValueError(f'Source checksum differs: {source}')
            verified_sources.add(source)
        destination = output / (row['clip'] + '.yuv')
        if destination.exists():
            if sha(destination) != row['sha256']:
                raise ValueError(f'Existing input differs; preserved without overwriting: {destination}')
        else:
            partial = temporary / (row['clip'] + '.yuv.partial')
            if partial.exists():
                raise FileExistsError(partial)
            start = int(row['start_frame'])
            frames = int(row['frames'])
            select = f'select=between(n\\,{start}\\,{start + frames - 1})'
            command = [str(ffmpeg), '-v', 'error', '-nostdin', '-hwaccel', 'none',
                       '-threads', '1', '-i', str(source), '-vf', select,
                       '-vsync', '0', '-frames:v', str(frames), '-c:v', 'rawvideo',
                       '-threads', '1', '-pix_fmt', row['source_pix_fmt'], '-f', 'rawvideo', str(partial)]
            subprocess.run(command, check=True, cwd=workspace)
            expected_bytes = frames * int(row['width']) * int(row['height']) * 3 // 2
            if partial.stat().st_size != expected_bytes or sha(partial) != row['sha256']:
                raise ValueError(f'Decoded window differs from frozen input; retained for inspection: {partial}')
            partial.replace(destination)
        verified.append(row['clip'])
        print('Verified:', row['clip'], flush=True)
    report = {'verified_windows': verified, 'requested_windows': len(selected),
              'all_246_windows_verified_in_this_invocation': len(verified) == len(rows),
              'native_yuv420': True, 'resized': False, 'rgb_round_trip': False}
    (job / 'prepared_test_inputs.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Verified {len(verified)} / {len(rows)} frozen test windows.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--limit', type=int, default=0, help='0: all windows; positive value: input smoke check only')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    prepare(args.config, args.ffmpeg, args.limit)
