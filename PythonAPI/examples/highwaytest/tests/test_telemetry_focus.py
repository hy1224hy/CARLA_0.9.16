import tempfile
import json
import unittest
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from highwaytest.config import Config
from highwaytest.training.telemetry import Telemetry


class FocusedTelemetryTests(unittest.TestCase):
    def test_old_jsonl_replay_is_filtered_without_rewriting_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'metrics.jsonl'
            original=json.dumps(dict(round=1,team='A',round_end_steps=16,
                policy_loss=.2,message_mean_effect=99))+'\n'
            path.write_text(original,encoding='utf-8')
            telemetry=Telemetry(folder,Config())
            telemetry.close()
            events=EventAccumulator(str(Path(folder)/'tensorboard')).Reload()
            self.assertEqual(path.read_text(encoding='utf-8'),original)
            self.assertEqual(events.Tags()['scalars'],['train/A/policy_loss'])
            self.assertEqual(events.Scalars('train/A/policy_loss')[0].step,16)

    def test_only_relevant_metrics_and_no_duplicate_family_panels(self):
        with tempfile.TemporaryDirectory() as folder:
            telemetry = Telemetry(folder, Config(group='G1'))
            telemetry.round([dict(team='A', round_end_steps=32, policy_loss=.2,
                mutual_kl=.3, mutual_loss=.006, effective_beta=.02,
                message_mean_effect=99, unknown_diagnostic=123)],
                [dict(team='A', family='parallel', **{'return':1.,
                    'success':True,'collision':False,'offroad':False,
                    'episode_steps':100,'return_agent_0':2.,'return_agent_1':0.,
                    'bad':False,'blue_progress_m':180.,'red_delay_m':12.,
                    'qualified_event':True})],
                dict(train_steps=32,validation_steps=8,gates={'A':{'parallel':1}},
                    gate={'A':{'overall':{'quality':.2},'families':{}}},
                    selection={'A':{'overall':{'success_rate':1.,
                        'qualified_attack_utility':.1,'episodes':8,
                        'unknown_validation':42},'families':{'parallel':{'quality':.2}}}}))
            telemetry.close()
            events=EventAccumulator(str(Path(folder)/'tensorboard')).Reload()
            tags=events.Tags()['scalars']
            for tag in ('train/A/policy_loss','train/A/mutual_loss',
                        'train/A/effective_beta','rollout/A/return',
                        'validation/selection/A/success_rate',
                        'validation/selection/A/qualified_attack_utility'):
                self.assertIn(tag,tags)
            for tag in tags:
                self.assertNotIn('unknown',tag)
                self.assertNotIn('parallel',tag)
                self.assertNotIn('message_',tag)
                self.assertNotIn('return_agent',tag)
                self.assertFalse(tag.startswith('gate/'))
            self.assertEqual(events.Scalars('train/A/mutual_loss')[0].step,32)

    def test_sparse_episode_and_nonfinite_values_do_not_break_logging(self):
        with tempfile.TemporaryDirectory() as folder:
            telemetry=Telemetry(folder,Config())
            telemetry.round([dict(team='B',round_end_steps=8,
                                 policy_loss=float('nan'))],
                            [dict(team='B',**{'return':2.})])
            telemetry.close()
            events=EventAccumulator(str(Path(folder)/'tensorboard')).Reload()
            self.assertNotIn('train/B/policy_loss',events.Tags()['scalars'])
            self.assertAlmostEqual(events.Scalars('rollout/B/return')[0].value,2.)
