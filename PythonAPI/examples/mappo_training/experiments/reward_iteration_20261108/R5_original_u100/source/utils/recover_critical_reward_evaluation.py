"""One registered fresh evaluation attempt after a failed serial pilot stage.

Preserves the partial attempt, trained models, configuration and pinned sources.
The evaluator child acquires the shared CARLA lease; no training is repeated.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]


def preflight():
    from utils.mut_mappo_session import CarlaOwner
    from evaluation.evaluate_base_policy_c0 import _preflight_idle_world
    with CarlaOwner(purpose='read-only restart preflight'):
        return _preflight_idle_world(host='127.0.0.1', port=2000, expected_version='0.9.16')


def recover(root, group, attempt=1, reuse_idle_world=True):
    from training.checkpoint_protocol import atomic_write_json, sha256_file
    from utils.mut_mappo_session import RunSessionOwner, now
    control = root / 'control'
    stage = group + '_gate'
    marker = control / (stage + '_completed.json')
    with RunSessionOwner(root / 'pipeline'):
        if marker.exists():
            raise FileExistsError('successful evaluation already registered')
        state = json.loads((control / 'pipeline_status.json').read_text(encoding='utf-8'))
        prior_stage=stage if attempt==1 else stage+'_retry'+str(attempt-1)
        if state.get('status') != 'FAILED_REVIEW_REQUIRED' or state.get('stage') != prior_stage:
            raise ValueError('only the recorded failed gate stage may be retried')
        sources = json.loads((control / 'pipeline_sources.json').read_text(encoding='utf-8'))
        for path, digest in sources.items():
            if sha256_file(ROOT / path) != digest:
                raise ValueError('pinned source changed: ' + path)
        trained = json.loads((control / (group + '_train_completed.json')).read_text(encoding='utf-8'))
        for path, digest in trained['artifact_hashes'].items():
            if sha256_file(root / path) != digest:
                raise ValueError('trained evidence changed: ' + path)
        readiness=preflight()  # Do not register an evaluation against a dead/busy server.
        attempt_name='retry'+str(attempt)
        output = root / 'evaluations' / (stage + '_' + attempt_name)
        registration_path = control / (stage + '_' + attempt_name + '_recovery_registration.json')
        if output.exists() or registration_path.exists():
            raise FileExistsError('this recovery attempt already exists; preserve its evidence')
        command = list(state['stage_command'])
        if reuse_idle_world and '--reuse-idle-world' not in command:
            command.append('--reuse-idle-world')
        previous_output = Path(command[command.index('--output-dir') + 1])
        command[command.index('--output-dir') + 1] = str(output)
        token = uuid.uuid4().hex
        log = control / (stage + '_' + attempt_name + '_' + token[:10] + '.log')
        atomic_write_json(dict(registered_at=now(), previous_pipeline_state=state,
            previous_output=str(previous_output), fresh_output=str(output), readiness=readiness,
            config_sha256=sha256_file(root / 'configs' / (group + '_B0_s42.json')),
            checkpoint_sha256=sha256_file(root / 'runs' / group / 'latest.pt'),
            reuse_idle_world=reuse_idle_world,
            critical_evaluator_source_sha256=sha256_file(ROOT/'utils/evaluate_mut_mappo_critical.py'),
            recovery_source_sha256=sha256_file(Path(__file__)),
            retry_reason='Native client exited; partial cohort excluded from effectiveness comparison',
            failed_attempt_cost_scope='Committed case ticks are a lower bound; interrupted constructor/cleanup may be unrecorded',
            command=command), registration_path)
        state.update(status='RUNNING', worker_pid=os.getpid(), stage=stage + '_' + attempt_name,
                     pipeline_uuid=token, stage_command=command, stage_log=str(log), at=now())
        state.pop('error', None)
        state.pop('traceback', None)
        try:
            with log.open('x', encoding='utf-8') as stream:
                kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    env=dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8'), **kwargs)
                state['stage_pid'] = process.pid
                atomic_write_json(state, control / 'pipeline_status.json')
                code = process.wait()
            exited = dict(stage=stage, attempt=attempt_name, returncode=code, stage_pid=process.pid,
                          worker_pid=os.getpid(), pipeline_uuid=token, ended_at=now(), log=str(log))
            atomic_write_json(exited, control / (stage + '_' + attempt_name + '_exit.json'))
            if code:
                raise RuntimeError('fresh gate attempt failed; preserve both attempts and review ' + str(log))
            report = output / 'analysis/critical_report.json'
            manifest = json.loads((output / 'evaluations/eval_000/manifest.json').read_text(encoding='utf-8'))
            if manifest['status'] != 'COMPLETED' or manifest['completed_attempted_episodes'] < manifest['planned_episodes']:
                raise ValueError('incomplete evaluation cannot be registered as successful')
            exited['artifact_hashes'] = {str(report.relative_to(root)): sha256_file(report)}
            exited['evaluation_root'] = str(output.relative_to(root))
            exited['retained_failed_evaluation_root'] = str(previous_output.relative_to(root))
            exited['retained_failed_evaluation_roots'] = [str(p.relative_to(root)) for p in
                [root/'evaluations'/stage, *[root/'evaluations'/(stage+'_retry'+str(n)) for n in range(1,attempt)]]]
            atomic_write_json(exited, marker)
            state['completed_stages'].append(stage)
            state.update(status='COMPLETED_FOR_ANALYSIS', stage_pid=None, ended_at=now())
            atomic_write_json(state, control / 'pipeline_status.json')
            return state
        except BaseException as error:
            state.update(status='FAILED_REVIEW_REQUIRED', error=str(error), traceback=traceback.format_exc(), at=now())
            atomic_write_json(state, control / 'pipeline_status.json')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--group', choices=('R0', 'R1', 'R2'), required=True)
    parser.add_argument('--attempt', type=int, choices=(1,2,3), default=1)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--reuse-idle-world',action=argparse.BooleanOptionalAction,default=True)
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else ROOT / args.root
    result=preflight() if args.preflight_only else recover(root.resolve(), args.group, args.attempt,args.reuse_idle_world)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
