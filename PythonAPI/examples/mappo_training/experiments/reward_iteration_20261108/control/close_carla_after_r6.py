"""One-shot shutdown hook for the explicitly authorized R6 completion."""
import datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('D:/study/yan2/CARLA_0.9.16/PythonAPI/examples/mappo_training/experiments/reward_iteration_20261108')
VERSION=ROOT/'R6_counterfactual_u100';CONTROL=ROOT/'control';RUN=VERSION/'run'
sys.path.insert(0,str(VERSION/'source'))
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(d):
 from training.checkpoint_protocol import atomic_write_json
 atomic_write_json(dict(at=now(),hook_pid=os.getpid(),**d),CONTROL/'R6_shutdown_status.json')
def shutdown():
 from utils.mut_mappo_session import CarlaOwner
 manifest=read(RUN/'checkpoint_manifest.json')
 checkpoint=(RUN/manifest['checkpoint']).resolve()
 if RUN.resolve() not in checkpoint.parents:raise RuntimeError('checkpoint outside run')
 digest=hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(checkpoint.read_bytes()).hexdigest()
 if manifest['completed_round']!=100 or digest!=manifest['sha256']:raise RuntimeError('final checkpoint verification failed')
 with CarlaOwner('127.0.0.1',2000,VERSION,'user authorized shutdown after R6 u100'):
  script=r'''$ErrorActionPreference='Stop'
$identity=Get-Content -LiteralPath 'D:/study/yan2/CARLA_0.9.16/PythonAPI/examples/mappo_training/experiments/reward_iteration_20261108/control/R6_shutdown_carla_identity.json' -Raw | ConvertFrom-Json
$closed=@()
foreach($target in $identity){
 $actual=Get-CimInstance Win32_Process -Filter "ProcessId=$($target.ProcessId)" | Select-Object ProcessId,Name,CommandLine,CreationDate
 if($null -eq $actual){continue}
 if($actual.Name -ne $target.Name -or $actual.CommandLine -ne $target.CommandLine){throw 'CARLA process identity changed; refusing shutdown'}
 $expected=[datetimeoffset]::Parse($target.CreationDate).UtcDateTime
 if([Math]::Abs(($actual.CreationDate.ToUniversalTime()-$expected).TotalSeconds) -gt 0.01){throw 'PID reused; refusing shutdown'}
 $process=Get-Process -Id $actual.ProcessId
 $null=$process.CloseMainWindow()
 if(-not $process.WaitForExit(5000)){Stop-Process -Id $actual.ProcessId -Force}
 $closed+=$actual.ProcessId
}
@{closed_pids=$closed;remaining_recorded_pids=@($identity.ProcessId | Where-Object {Get-Process -Id $_ -ErrorAction SilentlyContinue})} | ConvertTo-Json -Compress'''
  p=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',script],capture_output=True,text=True,timeout=45,creationflags=subprocess.CREATE_NO_WINDOW)
  if p.returncode:raise RuntimeError(p.stderr.strip())
  result=json.loads(p.stdout)
  if result['remaining_recorded_pids']:raise RuntimeError('recorded CARLA process remains alive')
  write(dict(status='CARLA_CLOSED_AFTER_U100',completed_updates=100,verified_checkpoint=str(checkpoint),checkpoint_sha256=digest,result=result,independent_evaluation=False))
def main():
 if '--launch' in sys.argv:
  dispatch=CONTROL/'R6_shutdown_dispatch.json'
  if dispatch.exists():raise RuntimeError('shutdown hook already dispatched')
  with (CONTROL/'R6_shutdown_console.log').open('x',encoding='utf-8') as output:
   child=subprocess.Popen([sys.executable,'-B','-X','utf8','-u',str(Path(__file__).resolve())],cwd=VERSION,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
  dispatch.write_text(json.dumps(dict(at=now(),pid=child.pid,script=str(Path(__file__).resolve())),indent=2)+'\n',encoding='utf-8')
  print(json.dumps(dict(pid=child.pid,status='completion hook started')));return
 from utils.mut_mappo_session import RunSessionOwner
 with RunSessionOwner(CONTROL/'R6_shutdown_hook_lock'):
  last=None
  while True:
   state=read(CONTROL/'state.json');worker=read(CONTROL/'worker.json');session=read(RUN/'session/status.json');manifest=read(RUN/'checkpoint_manifest.json')
   if state.get('version')!=VERSION.name:write(dict(status='CANCELLED_VERSION_CHANGED'));return
   if state.get('status')=='FAILED_REVIEW_REQUIRED':write(dict(status='TRAINING_FAILED_NOT_SHUT_DOWN',error=state.get('error')));return
   if manifest['completed_round']!=last:
    last=manifest['completed_round'];write(dict(status='WAITING_FOR_COMPLETE_U100',completed_updates=last,automations_cancelled=['carla-20','carla-5']))
   if state.get('status')=='TRAINING_COMPLETED' and state.get('completed_updates')==100 and not worker.get('active') and worker.get('returncode')==0 and session.get('status')=='SEGMENT_COMPLETED' and session.get('completed_round')==100:
    shutdown();return
   if state.get('status')=='STOPPED_AT_SAFE_BOUNDARY':write(dict(status='STOPPED_BEFORE_U100_NOT_SHUT_DOWN',completed_updates=manifest['completed_round']));return
   time.sleep(15)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(dict(status='SHUTDOWN_HOOK_FAILED_REVIEW_REQUIRED',error=repr(e)));raise
