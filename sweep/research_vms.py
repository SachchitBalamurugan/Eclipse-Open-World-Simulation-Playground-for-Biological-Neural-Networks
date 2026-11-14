"""Deploy and collect the surgical research suite on existing Freestyle workers.

Uses the CLI's stored login; never packages credentials or private challenge data.
"""
from pathlib import Path
import argparse
import concurrent.futures
import json
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
CLI = ['npx.cmd', '-y', 'freestyle@latest']

def fs(*args):
    return subprocess.run(CLI + list(args), cwd=ROOT, check=True, capture_output=True, text=True).stdout

def workers():
    return [line.split() for line in (ROOT/'sweep/jobs_state.txt').read_text().splitlines() if line.strip()]

def deploy(episodes):
    payload = ROOT/'sweep/collected/research_payload.tgz'
    payload.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(payload, 'w:gz') as tf:
        files = list((ROOT/'surgical').glob('*.py')) + list((ROOT/'surgical/model').rglob('*'))
        files += list((ROOT/'neurorover/culture').glob('*.py')) + [ROOT/'neurorover/__init__.py']
        files += [ROOT/'surgical/results'/f'culture_{mode}.pkl' for mode in ('teach','frozen')]
        for p in files:
            if p.is_file(): tf.add(p, arcname=p.relative_to(ROOT).as_posix())
    def start(item):
        i, (slug, vm) = item
        fs('vm','scp',payload.relative_to(ROOT).as_posix(),f'{vm}:/home/ubuntu/research_payload.tgz')
        # npx.cmd routes through cmd.exe, which mangles & > < in exec args, so ship scripts instead.
        job = ROOT/f'sweep/collected/research_job_{i}.sh'
        job.write_text('\n'.join([
            'rm -f ~/research_exit',
            'cd ~/nr && tar -xzf ~/research_payload.tgz',
            '. .v/bin/activate',
            f'python -m surgical.research_eval --seed {i+100} --episodes {episodes} '
            f'--out surgical/results/research_vm_{i}.json',
            'echo $? > ~/research_exit', '']), newline='\n')
        launcher = ROOT/f'sweep/collected/research_launch_{i}.sh'
        launcher.write_text('rm -f ~/research_exit\nnohup bash ~/research_job.sh > ~/research.log 2>&1 < /dev/null &\necho started\n', newline='\n')
        fs('vm','scp',job.relative_to(ROOT).as_posix(),f'{vm}:/home/ubuntu/research_job.sh')
        fs('vm','scp',launcher.relative_to(ROOT).as_posix(),f'{vm}:/home/ubuntu/research_launch.sh')
        fs('vm','exec',vm,'--','bash','/home/ubuntu/research_launch.sh')
        return {'slug':slug,'vm':vm,'seed':i+100,'output':f'research_vm_{i}.json'}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        manifest=list(pool.map(start, enumerate(workers())))
    (ROOT/'sweep/collected/research_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))

def collect():
    manifest=json.loads((ROOT/'sweep/collected/research_manifest.json').read_text())
    for row in manifest:
        vm=row['vm']
        status=fs('vm','exec',vm,'--','bash','-lc','if test -f ~/research_exit; then cat ~/research_exit; else echo RUNNING; fi; tail -n 2 ~/research.log')
        print(row['slug'],status.strip(),flush=True)
        if status.splitlines()[0].strip() == '0':
            fs('vm','scp',f'{vm}:/home/ubuntu/nr/surgical/results/{row["output"]}',f'surgical/results/{row["output"]}')

if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['deploy','collect']); ap.add_argument('--episodes',type=int,default=3)
    args=ap.parse_args()
    deploy(args.episodes) if args.action=='deploy' else collect()
