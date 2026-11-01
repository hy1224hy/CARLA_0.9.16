import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from utils.prepare_critical_reward_server import prepare

class ServerPrepareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_mut_mappo_runner import RunnerIntegrationTests as fixture
        fixture.setUpClass();cls.addClassCleanup(fixture.doClassCleanups)
        cls.red=Path(fixture.config['red_checkpoint']['path'])
        cls.digest=fixture.config['red_checkpoint']['sha256']
    def test_fresh_config_uses_parameters_and_needs_no_local_history(self):
        with tempfile.TemporaryDirectory() as tmp, patch('carla.Client',side_effect=AssertionError('offline preparation')):
            out=Path(tmp)/'experiment'
            record=prepare(self.red,out,device='cpu',seed=77,train_ticks=1234,host='192.0.2.1',port=2100,expected_sha256=self.digest)
            cfg=json.loads((out/'config.json').read_text())
            self.assertEqual(record['status'],'PREPARED_NOT_TRAINED')
            self.assertFalse(record['carla_connected'])
            self.assertEqual(cfg['task']['reward_profile'],'normal_driving_critical_team_v2')
            self.assertEqual(cfg['task']['c1_controller'],'lane_pid_v1')
            self.assertEqual(cfg['seed'],77);self.assertEqual(cfg['port'],2100)
            self.assertEqual(cfg['run']['train_tick_budget'],1234)
            self.assertEqual(cfg['device'],'cpu')
            self.assertEqual(Path(cfg['task']['team_reward']['reference_dir']),out/'references')
            self.assertEqual(Path(cfg['red_checkpoint']['path']),self.red.resolve())
            self.assertNotIn('diagnostic_authorization',cfg)
            with self.assertRaises(FileExistsError):prepare(self.red,out,device='cpu')
    def test_sha_mismatch_and_missing_sidecar_leave_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'out'
            with self.assertRaises(ValueError):prepare(self.red,out,expected_sha256='0'*64)
            self.assertFalse(out.exists())
            red=Path(tmp)/'uploaded.pt';red.write_bytes(self.red.read_bytes())
            with self.assertRaises(Exception):prepare(red,out)
            self.assertFalse(out.exists())
            self.assertFalse(Path(str(red)+'.sha256').exists())
    def test_overrides_are_configurable(self):
        from training.mut_mappo_config import ROOT
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);template=ROOT/'configs/mut_mappo/critical_reward_v2_server.json'
            source=root/'template.json';source.write_bytes(template.read_bytes())
            prepare(self.red,root/'out',source_config=source,device='cpu',
                case_manifest='data/mut_mappo/interaction_small_v4/cases.json',
                reference_dir=root/'custom_references',c1_controller='frozen_actor')
            cfg=json.loads((root/'out/config.json').read_text())
            self.assertEqual(cfg['task']['c1_controller'],'frozen_actor')
            self.assertEqual(Path(cfg['task']['team_reward']['reference_dir']),root/'custom_references')
    def test_endpoint_locks_are_configurable_and_default_compatible(self):
        from utils.mut_mappo_session import CarlaOwner
        self.assertEqual(CarlaOwner().path.name,'codex_carla_127_0_0_1_2000.lock')
        self.assertEqual(CarlaOwner('localhost').path,CarlaOwner().path)
        self.assertNotEqual(CarlaOwner(port=2100).path,CarlaOwner().path)
        self.assertNotEqual(CarlaOwner('192.0.2.1').path,CarlaOwner().path)
        for port in (True,0,65536):
            with self.assertRaises(ValueError):CarlaOwner(port=port)

    def test_invalid_budget_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'out'
            for budget in (0,-1):
                with self.assertRaises(ValueError):prepare(self.red,out,train_ticks=budget)
            self.assertFalse(out.exists())

if __name__=='__main__':unittest.main()
