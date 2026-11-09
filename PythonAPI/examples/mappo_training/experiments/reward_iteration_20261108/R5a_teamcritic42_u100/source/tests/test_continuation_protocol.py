import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from training.checkpoint_protocol import CheckpointProtocolError, sha256_file
from training.continuation_protocol import load_continuation, validate_continuation_checkpoint


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "training/continuation_protocol.py"
        self.source.parent.mkdir()
        self.source.write_text("pinned source")
        pins = {"training/continuation_protocol.py":sha256_file(self.source)}
        self.parent = {"sha256":"contract", "payload":{"planned_total_updates":100,
            "trainer_args":{"lr_decay_horizon":600,"entropy_anneal_updates":600},
            "critical_source_paths":[]}}
        self.payload = {"version":"r8c1_budget_continuation_v1", "id":"extension",
            "parent_contract_sha256":"contract", "run_id":"run", "parent_update":100,
            "planned_total_updates":600,"source_sha256":pins,"parent_source_sha256":{},
            "allowed_changed_sources":[],"parent_checkpoint_sha256":"checkpoint",
            "parent_code_fingerprint":"old","parent_git_sha":"git"}
        self.args = SimpleNamespace(continuation_contract="extension.json",resume="latest",
            pretrained_checkpoint="",allow_resume_code_drift=False,run_id="run",updates=600)
        self.checkpoint = {"update":100,"run_metadata":{"code_fingerprint":"old","last_git_sha":"git"}}

    def load(self):
        (self.root / "extension.json").write_text(json.dumps(self.payload))
        return load_continuation(self.args,self.parent,self.root)

    def validate(self, record, checkpoint=None, digest="checkpoint"):
        return validate_continuation_checkpoint(record,checkpoint or self.checkpoint,digest,
            "new","git",[self.source],self.root)

    def test_exact_parent_then_strict_resume(self):
        record=self.load()
        identity=self.validate(record)
        next_checkpoint={"update":101,"run_metadata":{"continuation":identity,
            "code_fingerprint":"new","last_git_sha":"git"}}
        self.assertEqual(self.validate(record,next_checkpoint,"next")["planned_total_updates"],600)

    def test_wrong_parent_digest(self):
        with self.assertRaises(CheckpointProtocolError): self.validate(self.load(),digest="wrong")

    def test_wrong_parent_update(self):
        self.checkpoint["update"]=99
        with self.assertRaises(CheckpointProtocolError): self.validate(self.load())

    def test_reject_source_edit(self):
        self.source.write_text("changed")
        with self.assertRaises(CheckpointProtocolError): self.load()

    def test_reject_source_omission(self):
        self.payload["source_sha256"]={}
        with self.assertRaises(CheckpointProtocolError): self.load()

    def test_reject_budget_overrun(self):
        self.args.updates=601
        with self.assertRaises(CheckpointProtocolError): self.load()

    def test_reject_schedule_reset(self):
        self.parent["payload"]["trainer_args"]["lr_decay_horizon"]=100
        with self.assertRaises(CheckpointProtocolError): self.load()

    def test_reject_generic_override(self):
        self.args.allow_resume_code_drift=True
        with self.assertRaises(CheckpointProtocolError): self.load()

    def test_reject_fresh(self):
        self.args.resume=""
        with self.assertRaises(CheckpointProtocolError): self.load()

    def test_reject_changed_extension_on_next_resume(self):
        record=self.load()
        identity=self.validate(record)
        identity["sha256"]="changed"
        c={"update":101,"run_metadata":{"continuation":identity,"code_fingerprint":"new","last_git_sha":"git"}}
        with self.assertRaises(CheckpointProtocolError): self.validate(record,c)

    def test_reject_scientific_source_change(self):
        name="environment/reward.py"
        source=self.root/name
        source.parent.mkdir()
        source.write_text("new reward")
        self.payload["source_sha256"][name]=sha256_file(source)
        self.payload["parent_source_sha256"][name]="old reward"
        self.payload["allowed_changed_sources"]=[name]
        with self.assertRaises(CheckpointProtocolError): self.load()


if __name__ == "__main__": unittest.main()
