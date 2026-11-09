"""Serial R4 pilot; a failed stage stops the sequence, without restarts."""
from pathlib import Path
import json,os,subprocess,sys,time,hashlib,shutil
from datetime import datetime
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CONTROL=HERE.parent/'control'

def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2),encoding='utf-8');os.replace(temp,path)

def stage(name,module,args):
    command=[sys.executable,'-X','utf8','-u','-m',module,*map(str,args)]
    state=dict(at=datetime.now().isoformat(),status='RUNNING',version=HERE.name,phase=name,command=command)
    write(CONTROL/'state.json',state)
    with (HERE/(name+'_console.log')).open('w',encoding='utf-8') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        write(CONTROL/'worker.json',dict(pid=child.pid,at=datetime.now().isoformat(),command=command))
        while child.poll() is None:time.sleep(5)
    if child.returncode:raise RuntimeError(name+' failed: '+str(child.returncode))

try:
    stage('validation','training.train_mut_mappo',['--config',HERE/'config.json','--validate-only'])
    stage('training','training.train_mut_mappo',['--config',HERE/'config.json','--run-dir',HERE/'run','--allow-candidate'])
    status=json.loads((HERE/'run/session/status.json').read_text())
    if status.get('completed_round')!=66:raise RuntimeError('R4 did not reach66 updates; inspect stop request, do not continue evaluation')
    for label,checkpoint in [('base_u0',next((HERE/'run/boundaries').glob('round_00000000_*.pt'))),('candidate_u66',HERE/'run/latest.pt')]:
        stage('evaluation_'+label,'evaluation.evaluate_mut_mappo',['--config',HERE/'config.json','--checkpoint',checkpoint,'--team','both','--split','selection','--conditions','C2','--output-dir',HERE/('evaluation_'+label)])
    results={}
    for label in ['base_u0','candidate_u66']:
        records=[json.loads(l) for l in (HERE/('evaluation_'+label)/'cases.jsonl').read_text().splitlines()]
        results[label]={}
        for team in ['A','B']:
            rs=[r for r in records if r['team']==team and r['valid']]
            results[label][team]={'n':len(rs),'bad_rate':sum(not r['blue_eligible'] for r in rs)/len(rs),'roles':{}}
            for aid in ['2','3']:
                ms=[r['individual_risk_metrics'][aid] for r in rs]
                results[label][team]['roles'][aid]={k:sum(m[k] for m in ms)/len(ms) for k in ['ttc_duration_s','ttc_longest_s','spacing_duration_s','spacing_longest_s','risk_integral_s','response_excess_integral_s','hard_brake_duration_s']}
    write(HERE/'analysis.json',dict(protocol='r4_risk_v1',results=results,claim='descriptive single-seed fixed-selection comparison; not proof of causal effect or success'))
    write(CONTROL/'state.json',dict(at=datetime.now().isoformat(),status='R4_TRIAL_COMPLETED_REVIEW_REQUIRED',version=HERE.name,phase='analysis',next='Review three dimensions and collision-conditioned results; do not apply legacy R3 selection gate or claim success without heldout validation'))
except BaseException as exc:
    write(CONTROL/'state.json',dict(at=datetime.now().isoformat(),status='FAILED',version=HERE.name,phase='stopped',error=repr(exc)))
    raise
