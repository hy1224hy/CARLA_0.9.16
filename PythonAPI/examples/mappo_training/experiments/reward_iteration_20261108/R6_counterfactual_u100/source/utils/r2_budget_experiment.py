"""Registered R2/B0 72k run, immutable milestones and serial frozen evaluation."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'experiments/adversarial_u1000_R2_budget_20261104_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def records(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines(keepends=True)
            if line.endswith('\n') and line.strip()]


def prepare(root):
    from training.checkpoint_protocol import atomic_write_json, sha256_file
    from training.mut_mappo_config import load_config, portable_config, validate_config
    from environment.mut_mappo_efficiency_reward import reference_identity, ReferenceStore
    from evaluation.critical_interaction_report import generate_report
    from evaluation.critical_interaction_metrics import PROTOCOL_DEFAULT
    from evaluation.interaction_quality import contract
    old = ROOT / 'experiments/adversarial_u1000_critical_reward_20261101_v4'
    cfg = portable_config(load_config(old / 'configs/R2_B0_s42.json'))
    if cfg['method'] != 'off' or cfg['task']['reward_profile'] != 'normal_driving_critical_team_v2':
        raise ValueError('expected the exact R2 B0 predecessor')
    cfg['run']['train_tick_budget'] = 72000
    cfg['task']['team_reward']['reference_dir'] = str(root / 'references/R2')
    cfg['diagnostic_authorization']['source'] = '2026-11-04 user authorized improved evaluation then registered R2 72k long experiment with monitoring'
    validated = validate_config(cfg)
    old_refs = old / 'references/R2'
    header = read(old_refs / 'manifest.json')
    if reference_identity(validated) != header['identity']:
        raise ValueError('reference task/red/cases identity differs')
    reference_store = ReferenceStore(old_refs, reference_identity(validated))
    for key in header['cases']:
        reference_store.curve(key)  # Verify every original SHA before any copy.
    root.mkdir(parents=True, exist_ok=False)
    control = root / 'control'
    control.mkdir()
    atomic_write_json(dict(status='PREPARING_OFFLINE', live_world_ticks=0), control / 'pipeline_status.json')
    (root/'references/R2').mkdir(parents=True)
    shutil.copyfile(old_refs/'manifest.json',root/'references/R2/manifest.json')
    for item in header['cases'].values():
        shutil.copyfile(old_refs/item['file'],root/'references/R2'/item['file'])
    atomic_write_json(dict(source=str(old_refs), manifest_sha256=sha256_file(old_refs/'manifest.json'),
        inherited_measured_episodes=144, new_reference_world_ticks=0,
        identity=header['identity'], curves_verified=len(header['cases'])), control/'reference_reuse.json')
    config_path = root / 'configs/R2_B0_s42.json'
    atomic_write_json(cfg, config_path)
    atomic_write_json(dict(allow_failed_C1_engineering_training=True, allow_weak_red_reference=True,
        red_frozen=True, formal_red_qualification=False, source=cfg['diagnostic_authorization']['source'],
        train_tick_budget=72000, frozen_evaluation_episodes=576), control/'training_authorization.json')
    atomic_write_json(contract(), control/'interaction_quality_contract.json')
    old_groups = [('R0', 'adversarial_u1000_critical_reward_20261101_v3'),
                  ('R1', 'adversarial_u1000_critical_reward_20261101_v3'),
                  ('R2', 'adversarial_u1000_critical_reward_20261101_v4')]
    reanalyses = {}
    for group, directory in old_groups:
        prior = ROOT / 'experiments' / directory
        marker = read(prior / 'control' / (group+'_gate_completed.json'))
        paths = [(p, digest) for p, digest in marker['artifact_hashes'].items()
                 if p.replace('\\', '/').endswith('/analysis/critical_report.json')]
        if len(paths) != 1:
            raise ValueError('expected one signed predecessor report')
        rel, digest = paths[0]
        prior_report = prior / rel.replace('\\', '/')
        if sha256_file(prior_report) != digest:
            raise ValueError('predecessor report SHA mismatch')
        source = Path(read(prior_report)['reports'][0]['source'])
        report = generate_report([source], root/'analysis'/('previous_'+group), PROTOCOL_DEFAULT)
        reanalyses[group] = dict(source_report_sha256=digest,
            output=str(root/'analysis'/('previous_'+group)/'critical_report.json'),
            team_rows=report['interaction_quality_team_rows'])
    atomic_write_json(reanalyses, control/'offline_reanalysis.json')
    sources = {str(p.relative_to(ROOT)).replace('\\', '/'): sha256_file(p)
        for folder in ('environment', 'training', 'evaluation', 'models') for p in (ROOT/folder).glob('*.py')}
    for name in ('utils/r2_budget_experiment.py', 'utils/evaluate_mut_mappo_critical.py',
                 'utils/mut_mappo_session.py', 'configs/evaluation/base_driving_vnext_v1.json'):
        sources[name] = sha256_file(ROOT/name)
    for name in sources:
        target = control/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    atomic_write_json(sources, control/'source_hashes.json')
    plan = dict(schema='r2_budget_experiment_v1', status='PREPARED', config=str(config_path),
        config_sha256=sha256_file(config_path), train_tick_budget=72000, milestone_ticks=[0,24000,48000,72000],
        frozen_episodes_per_node=144, frozen_episode_budget=576, evaluation_split='gate',
        initialization='same u1000 Actors; fresh four-car Critic/optimizer/ValueNorm',
        scope='one outer seed42, weak-red engineering only; A/B are not independent seeds',
        inherited_references='SHA-verified copy; no new CARLA sampling',
        internal_gate_schedule='10k warmup then20k train ticks; separate validation costs',
        environment=dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
        success_rule='joint observed contact-free resolved conflicts and full blue driving quality; paired C1 diagnostics; no return-only or formal superiority claim')
    atomic_write_json(plan, control/'experiment_plan.json')
    atomic_write_json(dict(status='PREPARED', live_world_ticks=0, next='run registered serial pipeline'),control/'pipeline_status.json')
    (control/'RESUME.md').write_text('R2 72k experiment, user authorized 2026-11-04. Read experiment_plan.json, pipeline_status.json and live_metrics.json first. Follow existing UUID/worker; never duplicate a controller. Sources are pinned in source_hashes.json. No reward/PPO/threshold changes during this run. Fixed weak red u1000, no base-policy retraining. Run as module from mappo_training. Failed-C1 engineering is authorized. All failed attempts and actual tick costs must be retained.\n', encoding='utf-8')
    return plan


def live_files(root, state):
    """File-only, refreshed by the existing pipeline worker every ten seconds."""
    from training.checkpoint_protocol import atomic_write_json
    from utils.mut_mappo_session import last_json_record, now
    run = root/'runs/R2'
    last = last_json_record(run/'metrics.jsonl') or {}
    panel = dict(at=now(), pipeline=state, completed_round=last.get('completed_round',0),
        actual_counters=last.get('actual_counters',{}), training_window=last.get('training_window',{}),
        source='completed A+B training boundaries; stochastic training is not frozen evaluation',
        evaluation_case_count=0)
    if state.get('evaluation_output'):
        panel['evaluation_case_count'] = len(records(Path(state['evaluation_output'])/'evaluations/eval_000/cases.jsonl'))
    panel['optimization'] = {team: {key: value for key, value in metric.items() if
        key.startswith(('approx_kl_', 'clip_frac_', 'grad_norm_', 'gate_fraction_', 'mutual_gated_kl_')) or
        key in ('explained_variance','critic_loss','preupdate_ratio_errors')}
        for team, metric in last.get('teams',{}).items()}
    window = [r for r in records(run/'metrics.jsonl') if 'training_window' in r][-12:]
    panel['rolling_12_rounds'] = {}
    for team in ('A','B'):
        samples = [r['training_window'][team] for r in window if team in r['training_window']]
        n = sum(r['n'] for r in samples)
        if n:
            panel['rolling_12_rounds'][team] = dict(n_episodes=n, **{key:
                sum(r[key]*r['n'] for r in samples)/n for key in ('b','safe_critical_event','red_success_rate')})
    atomic_write_json(panel, root/'control/live_metrics.json')
    page = '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="10"><title>R2 72k live metrics</title><style>body{font:15px sans-serif;margin:24px}pre{white-space:pre-wrap}</style><h1>R2 72k training and frozen evaluation</h1><p>File evidence updated every ten seconds. A/B are two learners, not two independent seeds.</p><pre>'+html.escape(json.dumps(panel,ensure_ascii=False,indent=2))+'</pre>'
    (root/'control/live_metrics.html').write_text(page,encoding='utf-8')


def milestone_boundaries(rows):
    output = []
    for target in (0,24000,48000,72000):
        candidates = [r for r in rows if (r.get('completed_round') == 0 if target == 0 else
            r.get('actual_counters',{}).get('train',0) >= target)]
        if not candidates:
            raise ValueError('missing committed milestone '+str(target))
        row = min(candidates,key=lambda r:r['completed_round'])
        output.append(dict(target_train_ticks=target, completed_round=row['completed_round'],
            actual_train_ticks=row.get('actual_counters',{}).get('train',0)))
    return output


def run(root):
    from training.checkpoint_protocol import atomic_write_json, sha256_file
    from utils.mut_mappo_session import RunSessionOwner, now
    control = root/'control'
    plan = read(control/'experiment_plan.json')
    def verify():
        if sha256_file(plan['config']) != plan['config_sha256']:
            raise ValueError('registered configuration changed')
        for path, digest in read(control/'source_hashes.json').items():
            if sha256_file(ROOT/path) != digest:
                raise ValueError('pinned source changed: '+path)
    state = dict(status='RUNNING', pipeline_uuid=uuid.uuid4().hex, worker_pid=os.getpid(),
                 started_at=now(), completed_stages=[])
    def stage(name, args, evaluation_output=None):
        verify()
        marker = control/(name+'_completed.json')
        if marker.exists():
            done = read(marker)
            for path, digest in done['artifact_hashes'].items():
                if sha256_file(root/path) != digest:
                    raise ValueError('completed evidence changed')
            state['completed_stages'].append(name)
            return
        log = control/(name+'_'+uuid.uuid4().hex[:10]+'.log')
        kwargs = dict(creationflags=subprocess.CREATE_NO_WINDOW) if os.name == 'nt' else {}
        env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYGAME_HIDE_SUPPORT_PROMPT='1',
                   **plan['environment'])
        with log.open('x',encoding='utf-8') as stream:
            process = subprocess.Popen([sys.executable,'-X','utf8','-u','-m',*args],cwd=ROOT,
                stdout=stream,stderr=subprocess.STDOUT,env=env,**kwargs)
            state.update(stage=name,stage_pid=process.pid,stage_log=str(log),at=now(),evaluation_output=str(evaluation_output) if evaluation_output else None)
            atomic_write_json(state, control/'pipeline_status.json')
            while process.poll() is None:
                live_files(root,state)
                time.sleep(10)
        exited = dict(stage=name,returncode=process.returncode,stage_pid=process.pid,
            worker_pid=os.getpid(),pipeline_uuid=state['pipeline_uuid'],ended_at=now(),log=str(log))
        atomic_write_json(exited,control/(name+'_exit.json'))
        if process.returncode:
            raise RuntimeError(name+' failed; inspect '+str(log))
        if name == 'train':
            session = root/'runs/R2/session'
            launch, end = read(session/'launch.json'), read(session/'exit.json')
            if any(launch[k] != end[k] for k in ('launch_id','trainer_pid','supervisor_pid')) or end['returncode']:
                raise ValueError('actual trainer exit identity differs')
            if not read(session/'status.json').get('training_budget_reached'):
                raise ValueError('trainer did not reach registered budget')
            exited['training_exit'] = end
            artifact = root/'runs/R2/checkpoint_manifest.json'
        else:
            manifest = read(evaluation_output/'evaluations/eval_000/manifest.json')
            report = read(evaluation_output/'analysis/critical_report.json')
            if manifest['status'] != 'COMPLETED' or manifest['completed_attempted_episodes'] != 144 or any(r.get('measurement_coverage') != 1 for r in report['team_rows']):
                raise ValueError('frozen evaluation coverage incomplete')
            artifact = evaluation_output/'analysis/critical_report.json'
        exited['artifact_hashes'] = {str(artifact.relative_to(root)):sha256_file(artifact)}
        atomic_write_json(exited,marker)
        state['completed_stages'].append(name)
    from utils.evaluate_mut_mappo_critical import gpu_memory_capture
    state['gpu_memory_path'] = str(control/('gpu_'+state['pipeline_uuid']+'.jsonl'))
    with RunSessionOwner(root/'pipeline'), gpu_memory_capture(Path(state['gpu_memory_path'])):
        try:
            verify()
            # Same shared ownership/idle check as the recovery path; no tick/reset.
            from utils.recover_critical_reward_evaluation import preflight
            atomic_write_json(preflight(),control/'server_readiness.json')
            args = ['utils.mut_mappo_session','supervise','--config',plan['config'],'--run-dir',str(root/'runs/R2')]
            if (root/'runs/R2/checkpoint_manifest.json').exists():
                args.append('--resume')
            stage('train',args)
            nodes = milestone_boundaries(records(root/'runs/R2/metrics.jsonl'))
            for node in nodes:
                paths = list((root/'runs/R2/boundaries').glob('round_'+format(node['completed_round'],'08d')+'_*.pt'))
                if len(paths) != 1:
                    raise ValueError('ambiguous immutable milestone')
                node.update(checkpoint=str(paths[0]),sha256=sha256_file(paths[0]))
            atomic_write_json(nodes,control/'milestones.json')
            for node in nodes:
                name = 'frozen_'+str(node['target_train_ticks']).zfill(6)
                out = root/'evaluations'/name
                if out.exists() and not (control/(name+'_completed.json')).exists():
                    raise FileExistsError('retain partial evaluation and explicitly register a new attempt: '+str(out))
                stage(name,['utils.evaluate_mut_mappo_critical','live','--config',plan['config'],
                    '--checkpoint',node['checkpoint'],'--split','gate','--output-dir',str(out),
                    '--monitor-mode','off','--reuse-idle-world'],out)
            comparison = []
            for node in nodes:
                report = read(root/'evaluations'/('frozen_'+str(node['target_train_ticks']).zfill(6))/'analysis/critical_report.json')
                for row in report['interaction_quality_team_rows']:
                    comparison.append(dict(**node,**row))
            atomic_write_json(dict(scope=plan['scope'],nodes=comparison),root/'analysis/budget_curve.json')
            state.update(status='COMPLETED_FOR_ANALYSIS',ended_at=now(),stage_pid=None)
        except BaseException as error:
            state.update(status='FAILED_REVIEW_REQUIRED',at=now(),error=str(error),traceback=traceback.format_exc())
            atomic_write_json(state,control/'pipeline_status.json')
            live_files(root,state)
            raise
        atomic_write_json(state,control/'pipeline_status.json')
        live_files(root,state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run'))
    parser.add_argument('--root',type=Path,default=DEFAULT)
    args = parser.parse_args()
    root = args.root.resolve()
    print(json.dumps(prepare(root) if args.command == 'prepare' else run(root),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
