"""Fresh bounded qualification and conditional authority probe; no rental shutdown."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/pusht-validated-20261102'


def run(args, name):
    with (BASE/name).open('xb') as log:
        result=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=log,stderr=log,timeout=14400)
    print(json.dumps(dict(stage=name,returncode=result.returncode,time=time.time())),flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


def main():
    assert socket.gethostname()=='c45db82630e6'
    if len(sys.argv)==1 or sys.argv[1:]==['recover-software-checks']:
        recovering = len(sys.argv)>1
        if recovering:
            assert (BASE/'launch.json').exists()
            assert not (BASE/'software-checks-passed').exists()
            assert not (BASE/'run-v1').exists()
            assert not (BASE/'preparation.log').exists()
        else:
            BASE.mkdir(exist_ok=False,parents=True)
        env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',
                 MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',GALLIUM_DRIVER='llvmpipe',
                 LP_NUM_THREADS='1',SDL_VIDEODRIVER='dummy',PYGAME_HIDE_SUPPORT_PROMPT='1')
        env.pop('PYTHONPATH',None)
        mode = 'execute-recovery' if recovering else 'execute'
        log_name = 'orchestration-recovery.log' if recovering else 'orchestration.log'
        with (BASE/log_name).open('xb') as log:
            child=subprocess.Popen([sys.executable,__file__,mode],cwd=ROOT,env=env,
                                   stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
        receipt = 'launch-recovery.json' if recovering else 'launch.json'
        (BASE/receipt).write_text(json.dumps(dict(pid=child.pid,started=time.time(),
                                                      rental_shutdown='manual'))+'\n')
        print(json.dumps(dict(pid=child.pid)),flush=True)
        return
    assert sys.argv[1:] in [['execute'], ['execute-recovery']]
    recovering = sys.argv[1]=='execute-recovery'
    run(['-m','pytest','-p','no:capture','-q','experiments/tests/test_pusht_authority.py',
         'experiments/tests/test_pusht_replay_bounds.py'],
        'software-checks-recovery.log' if recovering else 'software-checks.log')
    (BASE/'software-checks-passed').write_text('passed\n')
    for script, log in [('prepare_pusht_validated.py','preparation.log'),
                        ('run_pusht_validated.py','run.log'),
                        ('audit_pusht_validated.py','audit.log'),
                        ('preserve_pusht_validated.py','preservation.log')]:
        run(['experiments/scripts/'+script],log)
    (BASE/'pipeline-complete.json').write_text(json.dumps(dict(completed=time.time()))+'\n')


if __name__=='__main__':
    main()
