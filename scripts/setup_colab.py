"""Create a Python 3.11 runtime for the pinned Phi-V stack, independent of Colab's kernel."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

UV_VERSION = '0.9.5'


def run(*args):
    print('+', ' '.join(map(str,args)), flush=True)
    subprocess.run(list(map(str,args)),check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--venv',default='/content/urp-venv')
    p.add_argument('--bootstrap',default=f'/content/urp-bootstrap-uv-{UV_VERSION}')
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    bootstrap = Path(args.bootstrap)
    uv = bootstrap/'bin'/'uv'
    if not uv.is_file():
        run(sys.executable,'-m','pip','install','--only-binary=:all:',
            '--target',bootstrap,f'uv=={UV_VERSION}')
    python = Path(args.venv)/'bin'/'python'
    if not python.is_file():
        # uv downloads a managed interpreter when Python 3.11 is not on the machine.
        run(uv,'venv','--python','3.11',args.venv)
    version = subprocess.check_output([str(python),'-c','import sys; print(sys.version.split()[0])'],text=True).strip()
    if not version.startswith('3.11.'):
        raise RuntimeError(f'{python} uses {version}; choose a new --venv path for Python 3.11')
    run(uv,'pip','install','--python',python,'--only-binary',':all:',
        'torch==2.4.1','torchvision==0.19.1','--index-url','https://download.pytorch.org/whl/cu121')
    run(uv,'pip','install','--python',python,'--only-binary',':all:',
        '-r',repo/'requirements-colab.txt','torch==2.4.1','torchvision==0.19.1')
    run(uv,'pip','check','--python',python)
    run(python,'-c','import setuptools; print("setuptools:", setuptools.__version__)')
    print(json.dumps({'python':str(python),'version':version,'repository':str(repo)}),flush=True)
    print('Environment ready. Use this interpreter for every baseline command. No kernel restart needed.',flush=True)


if __name__ == '__main__':
    main()
