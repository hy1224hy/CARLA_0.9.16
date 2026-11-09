"""Start the authorized R5a critic-only repair from base u0, then serial evaluation."""
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

ROOT=Path('D:/study/yan2/CARLA_0.9.16/PythonAPI/examples/mappo_training/experiments/reward_iteration_20261108')
PROJECT=ROOT.parent.parent
VERSION=ROOT/'R5a_teamcritic42_u100'
MEMORY=ROOT/'control/teamcritic42_memory_observation.jsonl'
sys.path.insert(0,str(PROJECT))
spec=importlib.util.spec_from_file_location('teamcritic42_helpers',ROOT/'control/nightly_r5.py')
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)

def observe_memory(stop):
    script='Get-Process -Name CarlaUE4-Win64-Shipping -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,PrivateMemorySize64,WorkingSet64 | ConvertTo-Json -Compress'
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','available_phys','total_commit','available_commit','total_virtual','available_virtual','extended_virtual')]
    while not stop.is_set():
        try:
            p=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',script],capture_output=True,text=True,timeout=20,creationflags=subprocess.CREATE_NO_WINDOW)
            if p.returncode:raise RuntimeError(p.stderr.strip())
            raw=json.loads(p.stdout) if p.stdout.strip() else []
            m=Memory();m.length=ctypes.sizeof(m)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise OSError('GlobalMemoryStatusEx failed')
            record=dict(at=n.now(),carla=raw if isinstance(raw,list) else [raw],system_memory_load_percent=m.load,
                physical_total_bytes=m.total_phys,physical_available_bytes=m.available_phys,
                windows_total_pagefile_capacity_bytes=m.total_commit,windows_available_pagefile_capacity_bytes=m.available_commit)
        except Exception as error:record=dict(at=n.now(),error=repr(error))
        with MEMORY.open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        stop.wait(60)

def main():
    if '--launch' in sys.argv:
        if (VERSION/'run').exists() or (VERSION/'START_RECORD.json').exists():raise RuntimeError('experiment already started')
        dispatch=ROOT/'control/teamcritic42_dispatch_retry1.json'
        if dispatch.exists():raise RuntimeError('already dispatched; inspect before retry')
        log=ROOT/'control/teamcritic42_controller_retry1.log'
        command=[sys.executable,'-B','-X','utf8','-u',str(Path(__file__).resolve())]
        with log.open('x',encoding='utf-8') as output:
            child=subprocess.Popen(command,cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP,
                env={**os.environ,'PYTHONFAULTHANDLER':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        n.write(dispatch,dict(at=n.now(),pid=child.pid,command=command,log=str(log)))
        print(json.dumps(dict(pid=child.pid,log=str(log))))
        return
    from utils.mut_mappo_session import RunSessionOwner,CarlaOwner
    from training.mut_mappo_config import load_config,validate_config,run_identity
    from training.mut_mappo_checkpoint import CheckpointStore
    import carla
    import torch
    stop=threading.Event()
    with RunSessionOwner(ROOT/'control/controller_lock'):
        if n.read(ROOT/'control/worker.json').get('active'):raise RuntimeError('another worker is active')
        if (VERSION/'run').exists() or (VERSION/'START_RECORD.json').exists():raise RuntimeError('fresh u0 run required')
        cfg=validate_config(load_config(VERSION/'config.json'))
        assert cfg['learning']['critic_scope']=='blue_team' and cfg['learning']['critic_initialization']=='base_checkpoint'
        assert cfg['method']=='off' and cfg['learning']['beta']==0 and cfg['seed']==42 and cfg['run']['rounds']==cfg['run']['lr_horizon_rounds']==100
        hashes=n.read(VERSION/'source_hashes.json')
        for f,h in hashes.items():
            if hashlib.sha256((PROJECT/f).read_bytes()).hexdigest()!=h:raise RuntimeError('source changed since verification: '+f)
        archive=ROOT/'control/teamcritic42_previous_state_retry1'
        archive.mkdir(exist_ok=False)
        for name in ('state.json','worker.json'):shutil.copyfile(ROOT/'control'/name,archive/name)
        n.write(ROOT/'control/state.json',dict(at=n.now(),version=VERSION.name,status='RUNNING',phase='preflight',controller_pid=os.getpid(),target_updates=100))
        memory=threading.Thread(target=observe_memory,args=(stop,),daemon=True);memory.start()
        try:
            with CarlaOwner(cfg['host'],cfg['port'],VERSION/'preflight','team critic repair map check'):
                client=carla.Client(cfg['host'],cfg['port']);client.set_timeout(60.)
                if client.get_server_version()!='0.9.16':raise RuntimeError('server version mismatch')
                world=client.get_world();before=world.get_map().name;render=world.get_settings().no_rendering_mode
                changed=before.rsplit('/',1)[-1]!='Town05'
                if changed:world=client.load_world('Town05',reset_settings=False)
                actual=world.get_map().name
                if actual.rsplit('/',1)[-1]!='Town05' or world.get_settings().no_rendering_mode!=render:raise RuntimeError('map/render contract mismatch')
                n.write(VERSION/'START_RECORD.json',dict(at=n.now(),version=VERSION.name,controller_pid=os.getpid(),map_before=before,map_after=actual,
                    map_changed=changed,no_rendering_mode=render,fresh_base_initialization=True,resume=False,reward_changed=False,
                    critic_input_dim=42,critic_observation_dim=40,critic_local_id_dim=2,rollout_transport_dim=80,
                    base_checkpoint=cfg['red_checkpoint'],memory_log=str(MEMORY),source_files_verified=len(hashes)))
            shutil.copyfile(__file__,VERSION/'experiment_driver.py')
            n.write(VERSION/'driver_sha256.json',dict(sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
            n.execute(VERSION,'training_u100','training.train_mut_mappo',['--config',VERSION/'config.json','--run-dir',VERSION/'run','--allow-candidate'])
            if n.read(VERSION/'run/session/status.json').get('completed_round')!=100:raise RuntimeError('100 complete updates required')
            initial=n.initial_checkpoint(VERSION)
            baseline=n.evaluate(VERSION,initial,'evaluation_base_u0')
            candidate=n.evaluate(VERSION,VERSION/'run/latest.pt','evaluation_candidate_u100')
            effects=n.cf(VERSION,VERSION/'run/latest.pt')
            n.write(VERSION/'analysis.json',dict(at=n.now(),baseline=baseline,candidate=candidate,counterfactual=effects,
                selection_screen=any(x['engineering_screen_passed'] for x in effects.values()),
                purpose='Critic repair comparison at unchanged R5a reward; agent review required before further changes.'))
            n.write(ROOT/'control/state.json',dict(at=n.now(),version=VERSION.name,status='AWAITING_AGENT_REWARD_ANALYSIS',phase='review',completed_updates=100,automatic_parameter_changes=False))
        except BaseException as error:
            n.write(ROOT/'control/state.json',dict(at=n.now(),version=VERSION.name,status='FAILED_REVIEW_REQUIRED',error=repr(error),automatic_restart=False))
            raise
        finally:
            stop.set();memory.join(timeout=22)

if __name__=='__main__':main()
