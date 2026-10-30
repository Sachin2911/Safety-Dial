"""Extract the hash-verified public simulator archives into the pinned source layout."""

import json
from pathlib import Path
import shutil
import time
import zipfile

setup=Path('/workspace/setup-maniguard-v3-20261030')
datasets=Path('/workspace/Safety-Dial/third_party/ManiGuard/behavior-1k/datasets')
while not (setup/'assets.downloaded').exists():
    if (setup/'downloads.failed').exists():
        raise SystemExit('Public asset download failed.')
    time.sleep(5)
record=json.loads((setup/'assets.downloaded').read_text())
for filename,source in record['archives'].items():
    target_name='behavior-1k-assets' if filename.startswith('behavior-1k-assets') else 'omnigibson-robot-assets'
    destination=datasets/target_name
    destination.mkdir(parents=True,exist_ok=True)
    print('extracting',filename,flush=True)
    with zipfile.ZipFile(source['path']) as archive:
        archive.extractall(destination)
        print('extracted',target_name,len(archive.infolist()),flush=True)
while not (setup/'longfinger.downloaded').exists():
    time.sleep(5)
target=datasets/'omnigibson-robot-assets/models/franka/franka_panda_longfinger'
shutil.copytree(setup/'longfinger-staging',target,dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('.cache'))
assert (datasets/'behavior-1k-assets/VERSION').read_text().strip()=='3.7.2rc1'
assert (target/'usd').is_dir()
(setup/'assets-extracted.ready').write_text(json.dumps({'version':'3.7.2rc1','longfinger_ready':True})+'\n')
print('public simulator assets and long-finger robot ready',flush=True)
