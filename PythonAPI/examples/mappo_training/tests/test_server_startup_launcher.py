"""Exercise Bash launcher dispatch/refusal paths with fake system commands.

No screen server, conda environment, CARLA client or training process is started.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/run_server_startup_v1.sh'
if not SCRIPT.exists():  # The same test can run in the reviewed staging folder.
    SCRIPT = Path(__file__).with_name('run_server_startup_v1.sh')
BASH = shutil.which('bash')
if os.name == 'nt':
    BASH = r'D:\Program Files\git\bin\bash.exe' if Path(r'D:\Program Files\git\bin\bash.exe').exists() else None


@unittest.skipUnless(BASH, 'Bash is required')
class ServerStartupLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='startup_launcher_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / 'project with spaces'
        (self.project / 'scripts').mkdir(parents=True)
        self.script = self.project / 'scripts/run_server_startup_v1.sh'
        shutil.copyfile(SCRIPT, self.script)
        contract = self.project / 'configs/base_driving/base_driving_startup_v1_v1.json'
        contract.parent.mkdir(parents=True)
        contract.write_text('{}', encoding='utf-8')
        self.bin = self.root / 'fakebin'
        self.bin.mkdir()
        self.calls = self.root / 'calls.txt'
        self.cwd_file = self.root / 'cwd.txt'
        self.stub('screen', '''if [[ ${1:-} == -ls ]]; then
  [[ -z ${SCREEN_OCCUPIED:-} ]] || printf ' 123.%s (Detached)\n' "$SCREEN_OCCUPIED"
  exit 0
fi
printf 'screen\n' >> "$CALLS"
printf '%s\n' "$@" >> "$CALLS"
''')
        self.stub('ss', '[[ -z ${PORT_OCCUPIED:-} ]] || echo "LISTEN 0 128 0.0.0.0:2000"\nexit 0\n')
        self.stub('conda', 'pwd > "$CWD_FILE"\nprintf "conda\\n" >> "$CALLS"\nprintf "%s\\n" "$@" >> "$CALLS"\n')
        self.env = dict(os.environ, CONDA_BIN=(self.bin / 'conda').as_posix(),
                        CONDA_ENV='myconda', CALLS=self.calls.as_posix(), CWD_FILE=self.cwd_file.as_posix(),
                        PATH=str(self.bin) + os.pathsep + os.environ['PATH'])
        for name in ('SCREEN_OCCUPIED', 'PORT_OCCUPIED', 'MONITOR_PORT', 'TRAIN_SESSION'):
            self.env.pop(name, None)

    def stub(self, name, body):
        path = self.bin / name
        path.write_text('#!/usr/bin/env bash\nset -eu\n' + body, encoding='utf-8', newline='\n')
        path.chmod(0o755)

    def run_script(self, *args, **env):
        return subprocess.run([BASH, str(self.script), *args], cwd=self.root,
                              env=dict(self.env, **env), capture_output=True, text=True, timeout=20)

    def fresh(self):
        result = self.run_script('train')
        self.assertEqual(result.returncode, 0, result.stderr)
        run_id = re.search(r'BD_STARTUP_V1_s42_[0-9_]+', result.stdout).group()
        control = self.project / 'diagnostics/server_startup_v1' / run_id
        return run_id, control

    def test_help_never_dispatches(self):
        result = self.run_script('--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('resume RUN_ID', result.stdout)
        self.assertFalse(self.calls.exists())

    def test_train_preserves_paths_with_spaces_and_records_environment(self):
        _, control = self.fresh()
        argv = self.calls.read_text().splitlines()
        self.assertIn('-dmS', argv)
        self.assertIn('myconda', argv)
        self.assertIn('_worker', argv)
        self.assertEqual(argv[-1], 'fresh')
        self.assertTrue(any('project with spaces/scripts/run_server_startup_v1.sh' in arg for arg in argv))
        self.assertTrue((control / 'launcher.env').exists())
        self.assertFalse((control / 'session/plan.json').exists())
        self.assertFalse((self.project / 'logs').exists())

    def test_duplicate_training_screen_is_rejected_before_reserving_run(self):
        result = self.run_script('train', SCREEN_OCCUPIED='mappo-startup-v1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('already exists', result.stderr)
        self.assertFalse((self.project / 'diagnostics').exists())
        self.assertFalse(self.calls.exists())

    def test_existing_carla_ports_are_reused_without_launching_binary(self):
        result = self.run_script('carla', PORT_OCCUPIED='yes')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('not starting a duplicate', result.stdout)
        self.assertFalse(self.calls.exists())

    def test_run_id_cannot_escape_control_directory(self):
        result = self.run_script('resume', '../../other')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('original BD_STARTUP', result.stderr)
        self.assertFalse(self.calls.exists())

    def test_resume_without_prepared_plan_is_rejected(self):
        run_id, _ = self.fresh()
        self.calls.unlink()
        result = self.run_script('resume', run_id)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('No prepared plan', result.stderr)
        self.assertFalse(self.calls.exists())

    def test_resume_reuses_existing_run_and_saved_environment(self):
        run_id, control = self.fresh()
        plan = control / 'session/plan.json'
        plan.parent.mkdir()
        plan.write_text('EXISTING_PLAN', encoding='utf-8')
        self.calls.unlink()
        result = self.run_script('resume', run_id, CONDA_ENV='wrong-environment')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(plan.read_text(), 'EXISTING_PLAN')
        argv = self.calls.read_text().splitlines()
        self.assertIn('myconda', argv)
        self.assertEqual(argv[-1], 'resume')

    def test_status_uses_project_directory_from_an_unrelated_working_directory(self):
        run_id, control = self.fresh()
        plan = control / 'session/plan.json'
        plan.parent.mkdir()
        plan.write_text('{}', encoding='utf-8')
        result = self.run_script('status', run_id)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.cwd_file.read_text().strip().endswith('/project with spaces'))
        self.assertIn('utils.base_driving_session\ninspect\n--plan', self.calls.read_text())

    def test_invalid_monitor_port_prevents_dispatch(self):
        result = self.run_script('train', MONITOR_PORT='70000')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Invalid MONITOR_PORT', result.stderr)
        self.assertFalse(self.calls.exists())


if __name__ == '__main__':
    unittest.main()
