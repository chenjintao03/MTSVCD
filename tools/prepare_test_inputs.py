"""Extract native-YUV420 test windows and record input identities locally."""
import argparse
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
    split = json.loads((REPO / 'metadata/split_manifest.json').read_text(encoding='utf-8'))
    videos = sorted((r for r in split['records'] if r['subset'] == 'short' and r['split'] == 'test'),
                    key=lambda r: r['camera'])
    rows = []
    for video in videos:
        n = video['frames']
        for label, start in [('begin', 0), ('middle', (n - 96) // 2), ('end', n - 96)]:
            rows.append(dict(clip=video['camera'] + '_' + label, camera=video['camera'],
                             start_frame=start, frames=96, width=640, height=384,
                             source_pix_fmt=video['pix_fmt']))
    assert len(rows) == 246
    selected = rows[:limit] if limit else rows
    prior = {}
    for name in ('input_manifest.preview.json', 'input_manifest.json'):
        path = job / 'protocol' / name
        if path.is_file():
            prior.update({c['clip']: c for c in json.loads(path.read_text(encoding='utf-8'))['clips']})
    source_hashes = {}
    verified = []
    clips = []
    protocol = job / 'protocol'
    protocol.mkdir(parents=True, exist_ok=True)
    for row in selected:
        source = dataset / 'processed_video' / row['camera'] / 'clip01.mp4'
        if source not in source_hashes:
            source_hashes[source] = sha(source)
        source_digest = source_hashes[source]
        destination = output / (row['clip'] + '.yuv')
        if destination.exists():
            old = prior.get(row['clip'])
            if not old or old['source_sha256'] != source_digest or sha(destination) != old['sha256']:
                raise ValueError(f'Existing input cannot be verified against its local manifest; preserved: {destination}')
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
            if partial.stat().st_size != expected_bytes:
                raise ValueError(f'Decoded window has an unexpected size; retained for inspection: {partial}')
            partial.replace(destination)
        clips.append(dict(row, input=destination.as_posix(), source=source.as_posix(),
                          source_sha256=source_digest, sha256=sha(destination)))
        verified.append(row['clip'])
        preview = {'clips': clips, 'test_videos': 82, 'expected_windows': 246,
                   'complete': len(clips) == len(rows), 'identities_generated_locally': True}
        (protocol / 'input_manifest.preview.json').write_text(json.dumps(preview, indent=2) + '\n', encoding='utf-8')
        print('Verified:', row['clip'], flush=True)
    if len(clips) == len(rows):
        (protocol / 'input_manifest.json').write_text(json.dumps(preview, indent=2) + '\n', encoding='utf-8')
    report = {'verified_windows': verified, 'requested_windows': len(selected),
              'all_246_windows_verified_in_this_invocation': len(verified) == len(rows),
              'native_yuv420': True, 'resized': False, 'rgb_round_trip': False}
    (job / 'prepared_test_inputs.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Prepared {len(verified)} / {len(rows)} test windows; input identities are stored locally.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--limit', type=int, default=0, help='0: all windows; positive value: input smoke check only')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    prepare(args.config, args.ffmpeg, args.limit)
