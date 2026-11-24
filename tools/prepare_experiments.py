"""Render server experiment templates into a separate, explicitly configured workspace.

Only prepares files; never starts training, inference, or GPU jobs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

REPO = Path(__file__).resolve().parents[1]
TOKENS = ('WORKSPACE', 'DATASET_ROOT', 'VIMEO_ROOT', 'RT_WEIGHTS', 'TCM_WEIGHTS',
          'RT_PYTHON', 'TCM_PYTHON', 'GPU4_UUID', 'GPU5_UUID', 'REPO_ROOT',
          'UPSTREAM_ROOT', 'HOME')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def prepare(config):
    settings = json.loads(config.read_text(encoding='utf-8'))
    settings['REPO_ROOT'] = REPO.as_posix()
    settings.setdefault('HOME', Path.home().as_posix())
    settings.setdefault('UPSTREAM_ROOT', (REPO / 'Deep Learning-based').as_posix())
    missing = [key for key in TOKENS if not settings.get(key)]
    if missing:
        raise ValueError('Set these configuration keys: ' + ', '.join(missing))
    workspace = Path(settings['WORKSPACE']).resolve()
    if workspace.exists() and any(workspace.iterdir()):
        raise FileExistsError('Use a new empty workspace; existing experiments are never overwritten.')
    if workspace == REPO or REPO in workspace.parents:
        raise ValueError('The workspace must be outside the Git checkout.')
    workspace.mkdir(parents=True, exist_ok=True)
    for key in ('TEMP', 'TMP', 'TMPDIR', 'XDG_CACHE_HOME'):
        folder = workspace / ('cache' if key == 'XDG_CACHE_HOME' else 'tmp')
        folder.mkdir(exist_ok=True)
        os.environ[key] = str(folder)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    for p in sorted((REPO / 'experiments').rglob('*')):
        if not p.is_file():
            continue
        relative = p.relative_to(REPO / 'experiments')
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            text = p.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            shutil.copy2(p, target)
            continue
        for key in TOKENS:
            text = text.replace('@' + key + '@', str(settings[key]).replace('\\', '/'))
        unknown = re.findall(r'@(?:' + '|'.join(TOKENS) + r')@', text)
        assert not unknown, (relative, unknown)
        target.write_text(text, encoding='utf-8')
    rt = REPO / 'Deep Learning-based/DCVC-RT'
    tcm = REPO / 'Deep Learning-based/DCVC-TCM'
    for name in ('rt_matched', 'rt_endpoint', 'rt_selection', 'rt_original', 'rt_initial'):
        shutil.copytree(rt, workspace / name / 'implementation')
    shutil.copytree(tcm, workspace / 'tcm_matched/implementation')
    shutil.copytree(tcm, workspace / 'tcm_original/implementation/tcm')
    # Installed pytorch-msssim replaces the old, machine-local vendor directory.
    (workspace / 'rt_original/vendor').mkdir(parents=True, exist_ok=True)
    checkpoint_dir = workspace / 'rt_original/checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for name in ('cvpr2025_image.pth.tar', 'cvpr2025_video.pth.tar'):
        source = Path(settings['RT_WEIGHTS']) / name
        if source.is_file():
            os.symlink(source.resolve(), checkpoint_dir / name)
    manifest = json.loads((REPO / 'metadata/split_manifest.json').read_text(encoding='utf-8'))
    frames = {r['camera']: {'frames': r['frames'], 'pix_fmt': r['pix_fmt']}
              for r in manifest['records'] if r['subset'] == 'short'}
    for folder in ('rt_matched', 'tcm_matched', 'rt_endpoint', 'rt_selection'):
        protocol = workspace / folder / 'protocol'
        save(protocol / 'self_split.json', manifest)
        save(protocol / 'self_metadata.json', frames)
    # Refresh identities for the rendered files, retaining the release/source
    # mapping in Git. These are source fingerprints, never fabricated gate results.
    for p in workspace.glob('*/protocol/experiment.json'):
        value = json.loads(p.read_text(encoding='utf-8'))
        job = p.parent.parent
        for key, base in [('source_sha256', job), ('input_protocol_sha256', p.parent)]:
            if isinstance(value.get(key), dict):
                for name in list(value[key]):
                    candidate = base / name
                    if candidate.is_file():
                        value[key][name] = digest(candidate)
        save(p, value)
    for p in workspace.glob('*/protocol/source_sha256.json'):
        def refresh(value):
            if isinstance(value, dict):
                return {k: digest(Path(k)) if isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v)
                        and Path(k).is_file() else refresh(v) for k, v in value.items()}
            if isinstance(value, list):
                return [refresh(v) for v in value]
            return value
        save(p, refresh(json.loads(p.read_text(encoding='utf-8'))))
    save(workspace / 'tcm_matched/protocol/official_source_identity.json', {
        'original_sources': {p.as_posix(): digest(p) for p in sorted(tcm.rglob('*.py'))},
        'basis': 'This released server-source snapshot, with rendered workspace identities.'})
    configured = {key: str(settings[key]) for key in TOKENS}
    save(workspace / 'workspace_configuration.json', configured)
    print('Prepared:', workspace)
    print('No training, inference, GPU gate, or experiment controller was started.')
    print('Read docs/REPRODUCIBILITY.md before running the staged workflows.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    prepare(parser.parse_args().config)
