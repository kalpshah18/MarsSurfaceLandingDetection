"""Download the HiRISE DTM + orthoimage pairs listed in hirise_sites.json.

Files go to $HIRISE_DATA_DIR (default data/raw/hirise, git-ignored); each site gets its own
subdirectory. Downloads resume if interrupted (curl -C -). Several GB in total.

Usage:
    HIRISE_DATA_DIR=/path/outside/sync/folder python hirise_download.py
"""
import json
import os
import subprocess

DATA_DIR = os.environ.get('HIRISE_DATA_DIR', os.path.join('data', 'raw', 'hirise'))


def main():
    with open('hirise_sites.json') as f:
        sites = json.load(f)['sites']
    for s in sites:
        d = os.path.join(DATA_DIR, s['site'])
        os.makedirs(d, exist_ok=True)
        for key in ('dtm_url', 'ortho_url'):
            url = s[key]
            out = os.path.join(d, url.rsplit('/', 1)[1])
            print(f'{s["site"]}: {os.path.basename(out)}', flush=True)
            subprocess.run(['curl', '-s', '-S', '-f', '-L', '--retry', '5', '-C', '-', '-o', out, url], check=True)

    print('done')


if __name__ == '__main__':
    main()
