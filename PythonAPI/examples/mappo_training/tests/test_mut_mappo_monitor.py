"""Synthetic real-PPO/lifecycle checks; no CARLA or HTTP process is launched."""
import argparse
import copy
from contextlib import nullcontext
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from tests import test_mut_mappo_runner as fixtures
from training.mut_mappo_config import portable_config, run_identity
from training.mut_mappo_learning import TeamLearner
from training.train_mut_mappo import train
from visualization.mut_mappo_monitor import (
    MutMAPPOMonitor, add_monitor_arguments, monitor_cli_arguments, options_from_args,
)
from visualization.web_training_monitor import _LatestFrameStore, MonitorFrame


class MutMonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.RunnerIntegrationTests.setUpClass()
        cls.addClassCleanup(fixtures.RunnerIntegrationTests.doClassCleanups)
        cls.config = copy.deepcopy(fixtures.RunnerIntegrationTests.config)

    def monitor_factory(self):
        patcher = patch('visualization.web_training_monitor.WebTrainingMonitor')
        factory = patcher.start()
        factory.return_value.url = 'http://127.0.0.1:8080/'
        self.addCleanup(patcher.stop)
        return factory

    def test_cli_options_forward_and_do_not_change_scientific_identity(self):
        parser = argparse.ArgumentParser()
        add_monitor_arguments(parser)
        args = parser.parse_args(['--monitor-mode','web','--monitor-port','8081'])
        config = copy.deepcopy(self.config)
        original = run_identity(config)
        config['_monitor_options'] = options_from_args(args)
        self.assertEqual(original, run_identity(config))
        self.assertIn('8081', monitor_cli_arguments(args))
        self.assertEqual(monitor_cli_arguments(argparse.Namespace()), [])
        bad = parser.parse_args(['--monitor-fps','nan'])
        with self.assertRaises(ValueError):
            options_from_args(bad)

    def test_web_phases_and_gate_frames_preserve_policy_updates_and_ticks(self):
        factory = self.monitor_factory()
        config = copy.deepcopy(self.config)
        config['method'] = 'plain'
        config['mutual']['warmup_rounds'] = 1
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with patch('carla.Client', side_effect=AssertionError('no simulator')):
                plain = train(config, base/'off', env_factory=fixtures.RunnerIntegrationTests.factory)
                config['_monitor_options'] = {'mode':'web'}
                watched = train(config, base/'web', env_factory=fixtures.RunnerIntegrationTests.factory)
            self.assertEqual(plain['counters'], watched['counters'])
            for name in ('A','B'):
                for aid in (2,3):
                    for key, value in plain['teams'][name]['actors'][aid].items():
                        torch.testing.assert_close(value, watched['teams'][name]['actors'][aid][key], rtol=0, atol=0)
            web = factory.return_value
            phases = [call.args[0] for call in web.set_phase.call_args_list]
            for phase in ('rollout','optimizing','validating','saving','finalizing'):
                self.assertIn(phase, phases)
            self.assertEqual(factory.call_args.kwargs['view'], 'overhead')
            self.assertGreater(web.prepare_camera.call_count, 0)
            teams = {call.kwargs['team'] for call in web.begin_episode.call_args_list}
            self.assertEqual(teams, {'A','B'})
            self.assertEqual(web.finish.call_args.args[0], 'completed')
            self.assertEqual(web.finish.call_args.kwargs['result']['safe_round'], 2)
            web.close.assert_called_once()

    def test_partial_b_failure_reports_manifest_round_zero_and_cleans_camera_first(self):
        factory = self.monitor_factory()
        config = copy.deepcopy(self.config)
        config['_monitor_options'] = {'mode':'web'}
        events = []
        factory.return_value.stop_camera.side_effect = lambda: events.append('camera_stop')
        original_update, original_close = TeamLearner.update, fixtures.FakeLifecycleEnv.close
        def fail_b(team, *args, **kwargs):
            if team.name == 'B':
                raise RuntimeError('TEST_ONLY B update failure')
            return original_update(team, *args, **kwargs)
        def close(env):
            events.append('env_close')
            return original_close(env)
        with tempfile.TemporaryDirectory() as directory, patch.object(TeamLearner,'update',fail_b), \
             patch.object(fixtures.FakeLifecycleEnv,'close',close):
            with self.assertRaisesRegex(RuntimeError,'B update failure'):
                train(config, Path(directory)/'run', env_factory=fixtures.RunnerIntegrationTests.factory)
        self.assertLess(events.index('camera_stop'), events.index('env_close'))
        result = factory.return_value.finish.call_args
        self.assertEqual(result.args[0], 'error')
        self.assertEqual(result.kwargs['result']['safe_round'], 0)

    def test_independent_evaluation_has_frames_and_one_persistent_producer(self):
        from evaluation.evaluate_mut_mappo import evaluate
        factory = self.monitor_factory()
        config = copy.deepcopy(self.config)
        events = []
        original_close = fixtures.FakeLifecycleEnv.close
        factory.return_value.stop_camera.side_effect = lambda: events.append('camera_stop')
        def close(env):
            events.append('env_close')
            return original_close(env)
        def runtime(config, checkpoint, condition, team, audit_callback):
            result = fixtures.RunnerIntegrationTests.factory(config, audit_callback)
            result.condition, result.team = condition, team
            result.blue_actors = {2:result.red_actors[0], 3:result.red_actors[1]}
            return result
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            path.write_text(json.dumps(portable_config(config)),encoding='utf-8')
            with patch('utils.mut_mappo_session.CarlaOwner', return_value=nullcontext()), \
                 patch('training.train_mut_mappo.create_runtime', side_effect=runtime), \
                 patch.object(fixtures.FakeLifecycleEnv,'close',close), \
                 patch('carla.Client', side_effect=AssertionError('no simulator')):
                report = evaluate(path, team='A', conditions=('C1',),
                                  output_dir=Path(directory)/'evaluation', monitor_options={'mode':'web'})
        self.assertEqual(report['metadata']['status'],'COMPLETED')
        self.assertGreater(factory.return_value.submit.call_count,0)
        self.assertLess(events.index('camera_stop'),events.index('env_close'))
        self.assertEqual(factory.return_value.finish.call_args.kwargs['result']['purpose'],'evaluation')
        factory.return_value.close.assert_called_once()

    def test_offline_evaluation_does_not_start_monitor_even_when_requested(self):
        from evaluation.evaluate_mut_mappo import evaluate
        factory = self.monitor_factory()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            path.write_text(json.dumps(portable_config(self.config)),encoding='utf-8')
            result = evaluate(path, team='A', conditions=('C1',), validate_only=True,
                              monitor_options={'mode':'web'})
        self.assertFalse(result['carla_connected'])
        factory.assert_not_called()

    def test_frame_team_and_round_never_overwrite_live_phase(self):
        store = _LatestFrameStore('TEST_ONLY')
        generation = store.begin_episode({'round':1,'team':'A'})
        store.update_status({'round':2,'team':'B','phase':'optimizing'})
        frame = MonitorFrame({}, {'round':1,'team':'A','condition':'C2','monitor_kind':'mut_mappo'},0,generation)
        store.publish(b'TEST_ONLY_JPEG',frame,0)
        status = store.status()
        self.assertEqual((status['round'],status['team'],status['phase']),(2,'B','optimizing'))
        self.assertEqual((status['frame_round'],status['frame_team'],status['frame_condition']),(1,'A','C2'))


if __name__ == '__main__':
    unittest.main()
