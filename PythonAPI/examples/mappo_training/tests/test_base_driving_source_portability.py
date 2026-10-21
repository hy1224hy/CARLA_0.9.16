"""Verify frozen source provenance under Windows and Linux checkout layouts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from utils import build_base_driving_scenarios_v1 as builder
from utils.preflight_base_driving_scenarios_v1 import offline_validate


class BaseDrivingSourcePortabilityTests(unittest.TestCase):
    def test_source_identity_accepts_only_line_ending_equivalence(self):
        original = b'{\r\n  "coordinate": 1.25\r\n}\r\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.json'
            path.write_bytes(original)
            expected = hashlib.sha256(original).hexdigest()
            self.assertEqual(builder.frozen_source_sha256(path), expected)
            path.write_bytes(original.replace(b'\r\n', b'\n'))
            self.assertEqual(builder.frozen_source_sha256(path), expected)
            path.write_bytes(original.replace(b'1.25', b'1.26'))
            self.assertNotEqual(builder.frozen_source_sha256(path), expected)

    def test_all_frozen_payloads_match_both_source_checkout_styles(self):
        roots = (builder.DEFAULT_OUTPUT_DIR, builder.DEFAULT_VALIDATION_DIR)
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            source_dir = project / 'data/training_scenarios_town05_v2'
            source_dir.mkdir(parents=True)
            for crlf in (False, True):
                for source in builder.SOURCE_DIR.glob('scenario_*_free_flow.json'):
                    data = source.read_bytes().replace(b'\r\n', b'\n')
                    (source_dir / source.name).write_bytes(data.replace(b'\n', b'\r\n') if crlf else data)
                with self.subTest(crlf=crlf), patch.object(builder, 'PROJECT_DIR', project), \
                     patch.object(builder, 'SOURCE_DIR', source_dir):
                    for root in roots:
                        manifest, actual = offline_validate(root)
                        self.assertEqual(actual, builder.build_payloads(split=manifest['split']))
                        # This remains a byte-level check for the frozen artifacts.
                        for name, digest in manifest['scenario_sha256'].items():
                            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest)

    def test_rehashed_pool_cannot_hide_a_real_payload_change(self):
        manifest, payloads = offline_validate(builder.DEFAULT_OUTPUT_DIR)
        changed = deepcopy(payloads)
        name = next(iter(changed))
        # Metadata changes pass role geometry validation but must fail the
        # frozen generated-matrix comparison even with an updated manifest.
        changed[name]['description'] += ' modified'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'pool'
            shutil.copytree(builder.DEFAULT_OUTPUT_DIR, root)
            data = builder._serialize(changed[name]).encode('utf-8')
            (root / name).write_bytes(data)
            manifest['scenario_sha256'][name] = hashlib.sha256(data).hexdigest()
            (root / 'protocol/manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'pool differs'):
                offline_validate(root)


if __name__ == '__main__':
    unittest.main()
