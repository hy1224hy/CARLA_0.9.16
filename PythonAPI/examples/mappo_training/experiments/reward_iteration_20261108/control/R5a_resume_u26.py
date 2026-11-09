"""User-authorized continuation of the complete R5a u26 boundary, then original evaluations."""
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time

ROOT=Path(r'D:/study/yan2/CARLA_0.9.16/PythonAPI/examples/mappo_training/experiments/reward_iteration_20261108')
PROJECT=ROOT.parent.parent
VERSION=ROOT/'R5a_outcome10_u100'
sys.path.insert(0,str(PROJECT))
spec=importlib.util.spec_from_file_location('r5_resume_helper',ROOT/'control/nightly_r5.py')
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)

def memory_observer(stop):
    # Resource observation only; never restart or terminate the server.
    script="Get-Process -Name CarlaUE4-Win64-Shipping -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,PrivateMemorySize64,WorkingSet64 | ConvertTo-Json -Compress"
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_=[('dwLength',ctypes.c_ulong),('dwMemoryLoad',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('ullTotalPhys','ullAvailPhys','ullTotalPageFile','ullAvailPageFile','ullTotalVirtual','ullAvailVirtual','ullAvailExtendedVirtual')]
    while not stop.is_set():
        try:
            result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',script],capture_output=True,text=True,timeout=20)
            if result.returncode:raise RuntimeError(result.stderr.strip())
            raw=json.loads(result.stdout) if result.stdout.strip() else []
            processes=raw if isinstance(raw,list) else [raw]
            mem=MEMORYSTATUSEX();mem.dwLength=ctypes.sizeof(mem)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):raise OSError('GlobalMemoryStatusEx failed')
            record=dict(at=n.now(),carla=processes,system_memory_load_percent=mem.dwMemoryLoad,
                physical_total_bytes=mem.ullTotalPhys,physical_available_bytes=mem.ullAvailPhys,
                windows_total_pagefile_capacity_bytes=mem.ullTotalPageFile,windows_available_pagefile_capacity_bytes=mem.ullAvailPageFile)
        except Exception as error:record=dict(at=n.now(),error=repr(error))
        with (ROOT/'control/R5a_memory_observation.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')
        stop.wait(60)

def main():
    if '--launch' in sys.argv:
        command=[sys.executable,'-X','utf8','-u',str(Path(__file__).resolve())]
        log=ROOT/'control/R5a_resume_u26_controller.log'
        with log.open('x',encoding='utf-8') as output:
            process=subprocess.Popen(command,cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP,
                env={**os.environ,'PYTHONFAULTHANDLER':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        n.write(ROOT/'control/R5a_resume_dispatch.json',dict(at=n.now(),pid=process.pid,command=command,log=str(log)))
        print(json.dumps(dict(pid=process.pid,log=str(log))))
        return
    from utils.mut_mappo_session import RunSessionOwner
    from training.mut_mappo_config import load_config,validate_config,run_identity
    from training.mut_mappo_checkpoint import CheckpointStore
    stop=threading.Event()
    with RunSessionOwner(ROOT/'control/controller_lock'):
        state=n.read(ROOT/'control/state.json')
        if state.get('status')!='RESUME_WAITING_FOR_CARLA':raise RuntimeError('old worker cleanup has not been verified')
        cfg=validate_config(load_config(VERSION/'config.json'))
        store=CheckpointStore(VERSION/'run',run_identity(cfg))
        payload=store.load()
        if payload['completed_round']!=26:raise RuntimeError('expected complete u26 checkpoint')
        manifest=n.read(VERSION/'run/checkpoint_manifest.json')
        record=dict(at=n.now(),version=VERSION.name,checkpoint=manifest['checkpoint'],sha256=manifest['sha256'],
            completed_round=26,identity_verified=True,actor_critic_optimizer_rng_restored_by_common_trainer=True,
            reward_changed=False,lr_horizon=100,target_updates=100,controller_pid=os.getpid(),
            interruption='CARLA disappeared; Windows event 2004 at 10:55:30 reported virtual memory shortage, CARLA commit 50929131520 bytes',
            caveat='event strongly supports system commit exhaustion; no fresh engine crash stack recovered')
        n.write(VERSION/'RESUME_RECORD.json',record)
        shutil.copyfile(__file__,ROOT/'control/R5a_resume_u26.py')
        memory=threading.Thread(target=memory_observer,args=(stop,),daemon=True);memory.start()
        try:
            deadline=time.monotonic()+1200
            while True:
                try:
                    with socket.create_connection(('127.0.0.1',2000),timeout=2):break
                except OSError:
                    if time.monotonic()>deadline:raise RuntimeError('CARLA has not become reachable within 20 minutes; no training launched')
                    time.sleep(5)
            n.execute(VERSION,'training_resume_u26_to100','training.train_mut_mappo',
                ['--config',VERSION/'config.json','--run-dir',VERSION/'run','--allow-candidate','--resume'])
            if n.read(VERSION/'run/session/status.json').get('completed_round')!=100:raise RuntimeError('training did not reach u100')
            initial=n.initial_checkpoint(VERSION)
            baseline=n.evaluate(VERSION,initial,'evaluation_base_u0')
            checkpoint=VERSION/'run/latest.pt'
            candidate=n.evaluate(VERSION,checkpoint,'evaluation_candidate_u100')
            effects=n.cf(VERSION,checkpoint)
            n.write(VERSION/'analysis.json',dict(at=n.now(),baseline=baseline,candidate=candidate,counterfactual=effects,
                selection_screen=any(r['engineering_screen_passed'] for r in effects.values()),
                purpose='Agent review after complete evaluation; no automatic parameter selection.',resume_boundary=26))
            n.write(ROOT/'control/state.json',dict(at=n.now(),status='AWAITING_AGENT_REWARD_ANALYSIS',version=VERSION.name,
                phase='review',completed_updates=100,automatic_parameter_changes=False))
        except BaseException as error:
            n.write(ROOT/'control/state.json',dict(at=n.now(),status='FAILED_REVIEW_REQUIRED',version=VERSION.name,error=repr(error),automatic_restart=False))
            raise
        finally:
            stop.set();memory.join(timeout=22)

if __name__=='__main__':main()
