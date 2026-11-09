"""R6 training-only controller. No independent evaluation or automatic restart."""
import ctypes,datetime,hashlib,json,os,subprocess,sys,threading
from pathlib import Path
ROOT=Path('D:/study/yan2/CARLA_0.9.16/PythonAPI/examples/mappo_training/experiments/reward_iteration_20261108')
VERSION=ROOT/'R6_counterfactual_u100';SOURCE=VERSION/'source';MEMORY=ROOT/'control/R6_memory_observation.jsonl'
sys.path.insert(0,str(SOURCE))
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):
 from training.checkpoint_protocol import atomic_write_json
 atomic_write_json(d,p)
def observe(stop):
 class Memory(ctypes.Structure):
  _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','available_phys','total_commit','available_commit','total_virtual','available_virtual','extended_virtual')]
 while not stop.is_set():
  try:
   cmd='Get-Process -Name CarlaUE4-Win64-Shipping -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,PrivateMemorySize64,WorkingSet64 | ConvertTo-Json -Compress'
   p=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',cmd],capture_output=True,text=True,timeout=20,creationflags=subprocess.CREATE_NO_WINDOW)
   if p.returncode:raise RuntimeError(p.stderr)
   raw=json.loads(p.stdout or '[]');m=Memory();m.length=ctypes.sizeof(m)
   if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise OSError('memory query failed')
   record=dict(at=now(),carla=raw if isinstance(raw,list) else [raw],system_memory_load_percent=m.load,physical_total_bytes=m.total_phys,physical_available_bytes=m.available_phys,windows_total_pagefile_capacity_bytes=m.total_commit,windows_available_pagefile_capacity_bytes=m.available_commit)
  except Exception as e:record=dict(at=now(),error=repr(e))
  with MEMORY.open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
  stop.wait(60)
def main():
 if '--launch' in sys.argv:
  if (VERSION/'run').exists() or (VERSION/'START_RECORD.json').exists():raise RuntimeError('R6 already started: inspect before retry')
  dispatch=ROOT/'control/R6_dispatch.json'
  if dispatch.exists():raise RuntimeError('already dispatched')
  log=ROOT/'control/R6_controller.log';command=[sys.executable,'-B','-X','utf8','-u',str(Path(__file__).resolve())]
  with log.open('x',encoding='utf-8') as f:
   p=subprocess.Popen(command,cwd=SOURCE,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP,env={**os.environ,'PYTHONFAULTHANDLER':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
  write(dispatch,dict(at=now(),pid=p.pid,command=command,log=str(log)));print(json.dumps(dict(pid=p.pid,log=str(log))));return
 from utils.mut_mappo_session import CarlaOwner,RunSessionOwner
 from training.mut_mappo_config import load_config,validate_config
 import carla
 stop=threading.Event()
 with RunSessionOwner(ROOT/'control/controller_lock'):
  if read(ROOT/'control/worker.json').get('active'):raise RuntimeError('another worker is active')
  hashes=read(VERSION/'source_hashes.json')
  for name,digest in hashes.items():
   if hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('snapshot mismatch '+name)
  cfg=validate_config(load_config(VERSION/'config.json'))
  thread=threading.Thread(target=observe,args=(stop,),daemon=True);thread.start()
  try:
   with CarlaOwner(cfg['host'],cfg['port'],VERSION/'preflight','R6 training preflight'):
    client=carla.Client(cfg['host'],cfg['port']);client.set_timeout(90.)
    if client.get_server_version()!='0.9.16':raise RuntimeError('wrong server version')
    world=client.get_world();before=world.get_map().name;render=world.get_settings().no_rendering_mode
    if before.rsplit('/',1)[-1]!='Town05':world=client.load_world('Town05',reset_settings=False)
    if world.get_map().name.rsplit('/',1)[-1]!='Town05' or world.get_settings().no_rendering_mode!=render:raise RuntimeError('map/render mismatch')
    view=world.get_spectator().get_transform();view.rotation=carla.Rotation(pitch=-90.,yaw=0.,roll=0.);world.get_spectator().set_transform(view)
    write(VERSION/'START_RECORD.json',dict(at=now(),version=VERSION.name,controller_pid=os.getpid(),source_dir=str(SOURCE),source_files_verified=len(hashes),map_before=before,map_after=world.get_map().name,no_rendering_mode=render,spectator_pitch=-90.,follow_spectator=cfg['task']['follow_spectator'],target_updates=100,reward_profile=cfg['task']['reward_profile'],fresh_base_initialization=True,base_checkpoint=cfg['red_checkpoint'],critic_input_dim=42,critic_observation_dim=40,critic_local_id_dim=2,rollout_transport_dim=80,independent_evaluation_enabled=False,training_counterfactual_rollouts_per_case=2,memory_log=str(MEMORY),carla_restarted_for_memory=True))
   command=[sys.executable,'-B','-X','utf8','-u','-m','training.train_mut_mappo','--config',str(VERSION/'config.json'),'--run-dir',str(VERSION/'run'),'--allow-candidate']
   log=VERSION/'training_u100_console.log';state=dict(at=now(),version=VERSION.name,status='RUNNING',phase='training_u100',target_updates=100,controller_pid=os.getpid(),command=command,log=str(log),independent_evaluation_enabled=False)
   write(ROOT/'control/state.json',state)
   with log.open('x',encoding='utf-8') as f:
    child=subprocess.Popen(command,cwd=SOURCE,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,env={**os.environ,'PYTHONFAULTHANDLER':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    write(ROOT/'control/worker.json',dict(state,pid=child.pid,active=True));code=child.wait()
   write(ROOT/'control/worker.json',dict(state,pid=child.pid,active=False,returncode=code,ended_at=now()))
   if code:raise RuntimeError('R6 training exit '+str(code)+'; no automatic restart')
   status=read(VERSION/'run/session/status.json')
   write(ROOT/'control/state.json',dict(at=now(),version=VERSION.name,status='TRAINING_COMPLETED' if status['completed_round']==100 else 'STOPPED_AT_SAFE_BOUNDARY',phase='training_only_finished',completed_updates=status['completed_round'],independent_evaluation_enabled=False))
  except BaseException as e:
   write(ROOT/'control/state.json',dict(at=now(),version=VERSION.name,status='FAILED_REVIEW_REQUIRED',error=repr(e),automatic_restart=False));raise
  finally:stop.set();thread.join(timeout=22)
if __name__=='__main__':main()
