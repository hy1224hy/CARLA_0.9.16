"""Private model upload/config preparation; all weights are synthetic fixtures."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from models.actor_critic import Actor
from training.mut_mappo_config import ROOT, load_config, validate_config
from training.checkpoint_protocol import CheckpointProtocolError, checksum_path
from utils.prepare_mut_mappo_server import prepare


class PrepareMutServerTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'artifacts').mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix='TEST_ONLY_upload_', dir=ROOT/'artifacts')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.checkpoint = self.root / 'synthetic_red.pt'
        with torch.random.fork_rng():
            torch.manual_seed(81)
            actors = {aid: Actor(20, 2, hidden=16).state_dict() for aid in (0, 1)}
        self.payload = {'update': 7, 'model_spec': {'obs_dim':20, 'action_dim':2, 'hidden_dim':16},
                        'actors':actors, 'training_config': {'hidden_dim':16, 'gamma':.999,
                                                           'min_log_std':-2., 'max_log_std':0.}}
        self.save()
        self.output = self.root / 'generated'

    def save(self):
        torch.save(self.payload, self.checkpoint)
        self.digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()

    def test_upload_only_pt_is_verified_and_prepares_three_matched_groups_without_carla(self):
        self.assertFalse(checksum_path(self.checkpoint).exists())
        with patch('carla.Client', side_effect=AssertionError('must not connect')), \
             patch('subprocess.Popen', side_effect=AssertionError('must not dispatch')):
            result = prepare(self.checkpoint, self.output, self.digest, device='cpu')
        self.assertEqual(result['status'], 'PREPARED_NOT_TRAINED')
        self.assertEqual(result['source_update'], 7)
        self.assertFalse(result['carla_connected'])
        configs = [validate_config(load_config(self.output / f'{group}.json')) for group in ('b0','b1','b2')]
        self.assertEqual([c['method'] for c in configs], ['off','plain','selective'])
        self.assertEqual(len({c['_root_validation']['task_contract_sha256'] for c in configs}), 1)
        self.assertTrue(all(c['learning']['hidden_dim'] == 16 and c['run']['rounds'] == 2 for c in configs))
        self.assertTrue(all(c['_root_validation']['red_sha256'] == self.digest for c in configs))
        self.assertFalse(Path(configs[0]['red_checkpoint']['path']).is_absolute())
        self.assertTrue(checksum_path(self.checkpoint).exists())

    def test_wrong_upload_hash_creates_neither_sidecar_nor_config(self):
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            prepare(self.checkpoint, self.output, '0'*64)
        self.assertFalse(checksum_path(self.checkpoint).exists())
        self.assertFalse(self.output.exists())

    def test_existing_output_and_bad_sidecar_are_never_overwritten(self):
        self.output.mkdir()
        marker = self.output / 'evidence.txt'
        marker.write_text('preserve')
        with self.assertRaises(FileExistsError):
            prepare(self.checkpoint, self.output, self.digest)
        self.assertEqual(marker.read_text(), 'preserve')
        checksum_path(self.checkpoint).write_text('0'*64+'\n')
        with self.assertRaisesRegex(CheckpointProtocolError, 'SHA256 mismatch'):
            prepare(self.checkpoint, self.root/'other', self.digest)
        self.assertEqual(checksum_path(self.checkpoint).read_text(), '0'*64+'\n')

    def test_legacy_model_is_rejected_before_preparing_configs(self):
        self.payload['model_spec']['obs_dim'] = 19
        self.save()
        with self.assertRaisesRegex(ValueError, 'compatible 20/2'):
            prepare(self.checkpoint, self.output, self.digest)
        self.assertFalse(self.output.exists())

    def test_invalid_budget_is_rejected_before_registering_upload(self):
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            prepare(self.checkpoint, self.output, self.digest, rounds=0)
        self.assertFalse(checksum_path(self.checkpoint).exists())
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
