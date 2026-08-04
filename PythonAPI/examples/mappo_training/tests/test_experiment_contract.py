import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from training.experiment_contract import (
    EXPERIMENT_CONTRACT_VERSION,
    EXPERIMENT_MANIFEST_NAME,
    RESOLVED_CONFIG_NAME,
    ExperimentContractError,
    apply_experiment_contract,
    enforce_formal_transition_minimum,
    initialize_experiment_evidence,
    load_experiment_contract,
    update_experiment_evidence,
    validate_experiment_contract,
    validate_resume_experiment_evidence,
    verify_critical_sources_clean,
)
from training.train_no_rllib_mappo_robust_v2 import build_parser


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    PROJECT_DIR / "configs" / "base_mappo" / "p0_base_pilot_v1.json"
)


def _formal_args(save_dir: Path, *, updates: int = 5):
    run_id = save_dir.name
    argv = [
        "--experiment-contract",
        str(CONTRACT_PATH),
        "--updates",
        str(updates),
        "--run-id",
        run_id,
        "--save-dir",
        str(save_dir),
    ]
    args = build_parser().parse_args(argv)
    args._explicit_cli_options = {
        token.split("=", 1)[0] for token in argv if token.startswith("--")
    }
    return args


class ExperimentContractValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CONTRACT_PATH.open("r", encoding="utf-8") as handle:
            cls.payload = json.load(handle)

    def test_repository_contract_is_frozen_and_single_full_batch(self):
        record = load_experiment_contract(CONTRACT_PATH, project_dir=PROJECT_DIR)
        payload = record["payload"]
        self.assertEqual(payload["contract_version"], EXPERIMENT_CONTRACT_VERSION)
        self.assertEqual(payload["status"], "frozen")
        self.assertEqual(payload["trainer_args"]["ppo_epochs"], 5)
        self.assertEqual(payload["trainer_args"]["minibatch_size"], 32768)
        self.assertGreaterEqual(
            payload["trainer_args"]["minibatch_size"],
            2
            * payload["trainer_args"]["rollout_episodes"]
            * payload["trainer_args"]["max_steps"],
        )
        self.assertEqual(
            payload["trainer_args"]["lr_decay_horizon"],
            payload["planned_total_updates"],
        )
        self.assertEqual(
            payload["evaluation_contract"]["milestone_updates"],
            payload["segment_stop_updates"],
        )

    def test_unknown_fields_are_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload["silent_override"] = True
        with self.assertRaisesRegex(ExperimentContractError, "unknown"):
            validate_experiment_contract(payload, project_dir=PROJECT_DIR)

    def test_fragmented_minibatch_is_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload["trainer_args"]["minibatch_size"] = 256
        with self.assertRaisesRegex(ExperimentContractError, "single_full_batch"):
            validate_experiment_contract(payload, project_dir=PROJECT_DIR)

    def test_segment_must_align_with_safe_checkpoint(self):
        payload = copy.deepcopy(self.payload)
        payload["segment_stop_updates"] = [4, 10, 20, 30, 40, 60, 80, 100]
        payload["evaluation_contract"]["milestone_updates"] = list(
            payload["segment_stop_updates"]
        )
        with self.assertRaisesRegex(ExperimentContractError, "save_interval"):
            validate_experiment_contract(payload, project_dir=PROJECT_DIR)


class ExperimentContractApplicationTests(unittest.TestCase):
    def test_contract_is_authoritative_and_resolves_scenario_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir) / "P0_BASE_MAPPO_Town02_s42_20260804_210000"
            args = _formal_args(save_dir)
            metadata = apply_experiment_contract(args, project_dir=PROJECT_DIR)
            self.assertEqual(args.ppo_epochs, 5)
            self.assertEqual(args.minibatch_size, 32768)
            self.assertEqual(args.lr_decay_horizon, 100)
            self.assertEqual(Path(args.scenario_dir).resolve(), (
                PROJECT_DIR / "data" / "training_scenarios_town02_v2"
            ).resolve())
            self.assertEqual(args.config_path, str(CONTRACT_PATH.resolve()))
            self.assertEqual(metadata["requested_stop_update"], 5)

    def test_scientific_cli_override_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir) / "P0_BASE_MAPPO_Town02_s42_20260804_210000"
            args = _formal_args(save_dir)
            args._explicit_cli_options.add("--gamma")
            with self.assertRaisesRegex(ExperimentContractError, "unsupported CLI"):
                apply_experiment_contract(args, project_dir=PROJECT_DIR)

    def test_unregistered_stop_update_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir) / "P0_BASE_MAPPO_Town02_s42_20260804_210000"
            args = _formal_args(save_dir, updates=6)
            with self.assertRaisesRegex(ExperimentContractError, "not a registered"):
                apply_experiment_contract(args, project_dir=PROJECT_DIR)

    def test_run_directory_must_match_run_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir) / "P0_BASE_MAPPO_Town02_s42_20260804_210000"
            args = _formal_args(save_dir)
            args.save_dir = str(Path(temp_dir) / "different")
            with self.assertRaisesRegex(ExperimentContractError, "basename"):
                apply_experiment_contract(args, project_dir=PROJECT_DIR)

    def test_formal_low_transition_update_is_a_hold(self):
        formal_args = SimpleNamespace(
            _experiment_contract_metadata={"contract_id": "formal"},
            min_transitions_per_update=512,
        )
        with self.assertRaisesRegex(ExperimentContractError, "formal experiment HOLD"):
            enforce_formal_transition_minimum(
                args=formal_args, total_transitions=511
            )
        enforce_formal_transition_minimum(
            args=formal_args, total_transitions=512
        )
        enforce_formal_transition_minimum(
            args=SimpleNamespace(
                _experiment_contract_metadata=None,
                min_transitions_per_update=512,
            ),
            total_transitions=0,
        )

    @mock.patch("training.experiment_contract.subprocess.run")
    def test_dirty_critical_source_is_rejected(self, run_mock):
        run_mock.return_value = SimpleNamespace(
            returncode=0,
            stdout=" M PythonAPI/examples/mappo_training/training/experiment_contract.py\n",
            stderr="",
        )
        with self.assertRaisesRegex(ExperimentContractError, "committed critical"):
            verify_critical_sources_clean(
                repository_root=PROJECT_DIR.parents[2],
                project_dir=PROJECT_DIR,
                relative_paths=["training/experiment_contract.py"],
            )


class ExperimentEvidenceTests(unittest.TestCase):
    def test_evidence_and_registry_are_created_and_updated_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "P0_BASE_MAPPO_Town02_s42_20260804_210000"
            run_dir.mkdir()
            args = _formal_args(run_dir)
            metadata = apply_experiment_contract(args, project_dir=PROJECT_DIR)
            metadata["registry_path"] = str(root / "registry.csv")

            initialize_experiment_evidence(
                args=args,
                run_dir=run_dir,
                git_metadata={"commit": "abc123", "dirty": False, "dirty_paths": []},
                code_fingerprint="code-fingerprint",
                scenario_pool_fingerprint="scenario-fingerprint",
                training_config={"gamma": args.gamma, "ppo_epochs": args.ppo_epochs},
            )
            self.assertTrue((run_dir / RESOLVED_CONFIG_NAME).is_file())
            self.assertTrue((run_dir / EXPERIMENT_MANIFEST_NAME).is_file())
            validate_resume_experiment_evidence(args=args, run_dir=run_dir)

            update_experiment_evidence(
                args=args,
                run_dir=run_dir,
                completed_update=5,
                status="SEGMENT_COMPLETED",
            )
            with (run_dir / EXPERIMENT_MANIFEST_NAME).open(
                "r", encoding="utf-8"
            ) as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["completed_update"], 5)
            self.assertEqual(manifest["status"], "SEGMENT_COMPLETED")

            with (root / "registry.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["completed_update"], "5")
            self.assertEqual(rows[0]["status"], "SEGMENT_COMPLETED")

    def test_resume_rejects_contract_identity_drift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "P0_BASE_MAPPO_Town02_s42_20260804_210000"
            run_dir.mkdir()
            args = _formal_args(run_dir)
            metadata = apply_experiment_contract(args, project_dir=PROJECT_DIR)
            metadata["registry_path"] = str(root / "registry.csv")
            initialize_experiment_evidence(
                args=args,
                run_dir=run_dir,
                git_metadata={"commit": "abc123", "dirty": False, "dirty_paths": []},
                code_fingerprint="code-fingerprint",
                scenario_pool_fingerprint="scenario-fingerprint",
                training_config={"gamma": args.gamma},
            )
            args._experiment_contract_metadata["sha256"] = "different"
            with self.assertRaisesRegex(ExperimentContractError, "SHA256 mismatch"):
                validate_resume_experiment_evidence(args=args, run_dir=run_dir)


if __name__ == "__main__":
    unittest.main()
