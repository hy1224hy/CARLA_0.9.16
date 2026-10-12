import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from training.checkpoint_protocol import CheckpointProtocolError
from training.experiment_contract import (
    EXPERIMENT_CONTRACT_VERSION,
    EXPERIMENT_MANIFEST_NAME,
    RESOLVED_CONFIG_NAME,
    TRAINER_ARG_NAMES,
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
from training.train_no_rllib_mappo_robust_v2 import (
    build_parser,
    build_training_environment_kwargs,
    observation_protocol_version,
    validate_environment_reward_discount,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    PROJECT_DIR / "configs" / "base_mappo" / "p0_base_pilot_v1.json"
)
R1_CONTRACT_PATH = (
    PROJECT_DIR / "configs" / "base_mappo" / "p0_base_action_v2.json"
)
R2_CONTRACT_PATH = (
    PROJECT_DIR / "configs" / "base_mappo" / "p0_base_reward_v3.json"
)
R3_CONTRACT_PATH = (
    PROJECT_DIR / "configs" / "base_mappo" / "p0_base_critic_v4.json"
)
TOWN05_CONTRACT_PATH = (
    PROJECT_DIR / "configs" / "base_mappo" / "p0_base_town05_v1.json"
)
TOWN05_V2_CONTRACT_PATH = (
    PROJECT_DIR / "configs" / "base_mappo" / "p0_base_town05_v2.json"
)
TOWN05_V3_CONTRACT_PATH = (
    PROJECT_DIR / "configs" / "base_mappo" / "p0_base_town05_v3.json"
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

    def test_r1_contract_is_fresh_and_freezes_action_v2_without_reward_drift(self):
        record = load_experiment_contract(R1_CONTRACT_PATH, project_dir=PROJECT_DIR)
        payload = record["payload"]
        self.assertEqual(payload["contract_id"], "P0_BASE_MAPPO_TOWN02_ACTION_V2")
        self.assertEqual(payload["source_mode"], "fresh")
        self.assertEqual(payload["trainer_args"], self.payload["trainer_args"])
        self.assertEqual(
            payload["evaluation_contract"]["protocol_version"],
            "p0_base_c0_eval_v2",
        )
        self.assertIn("training/action_protocol.py", payload["critical_source_paths"])

    def test_r2_contract_changes_only_reward_and_freezes_reward_source(self):
        r1 = load_experiment_contract(R1_CONTRACT_PATH, project_dir=PROJECT_DIR)[
            "payload"
        ]
        r2 = load_experiment_contract(R2_CONTRACT_PATH, project_dir=PROJECT_DIR)[
            "payload"
        ]
        self.assertEqual(r2["contract_id"], "P0_BASE_MAPPO_TOWN02_REWARD_V3")
        self.assertEqual(r2["source_mode"], "fresh")
        r1_args = dict(r1["trainer_args"])
        r2_args = dict(r2["trainer_args"])
        self.assertEqual(r1_args.pop("reward_version"), "v2")
        self.assertEqual(r2_args.pop("reward_version"), "v3")
        self.assertEqual(r1_args, r2_args)
        self.assertEqual(
            r2["evaluation_contract"]["protocol_version"],
            "p0_base_c0_eval_v3",
        )
        self.assertIn(
            "environment/base_reward_v3.py",
            r2["critical_source_paths"],
        )

    def test_r3_contract_changes_only_critic_target_mechanism(self):
        r2 = load_experiment_contract(R2_CONTRACT_PATH, project_dir=PROJECT_DIR)[
            "payload"
        ]
        r3 = load_experiment_contract(R3_CONTRACT_PATH, project_dir=PROJECT_DIR)[
            "payload"
        ]
        self.assertEqual(r3["contract_version"], "p0_base_experiment_contract_v2")
        self.assertEqual(r3["contract_id"], "P0_BASE_MAPPO_TOWN02_CRITIC_V4")
        r3_args = dict(r3["trainer_args"])
        self.assertEqual(r3_args.pop("value_norm_beta"), 0.99999)
        self.assertEqual(r3_args.pop("value_norm_epsilon"), 1e-5)
        self.assertEqual(r3_args.pop("value_norm_variance_floor"), 1e-2)
        self.assertEqual(r3_args.pop("huber_delta"), 10.0)
        self.assertEqual(r3_args, r2["trainer_args"])
        self.assertIn("training/value_normalization.py", r3["critical_source_paths"])
        self.assertEqual(
            r3["evaluation_contract"]["protocol_version"],
            "p0_base_c0_eval_v4",
        )

    def test_unknown_fields_are_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload["silent_override"] = True
        with self.assertRaisesRegex(ExperimentContractError, "unknown"):
            validate_experiment_contract(payload, project_dir=PROJECT_DIR)

    def test_town05_v2_registers_comprehensive_curriculum_and_perception(self):
        record = load_experiment_contract(
            TOWN05_V2_CONTRACT_PATH,
            project_dir=PROJECT_DIR,
        )
        payload = record["payload"]
        self.assertEqual(
            payload["contract_id"],
            "P0_BASE_MAPPO_TOWN05_V2",
        )
        self.assertEqual(payload["planned_total_updates"], 160)
        self.assertEqual(
            payload["trainer_args"]["scenario_dir"],
            "data/training_scenarios_town05_v2",
        )
        self.assertEqual(payload["trainer_args"]["lidar_range_m"], 20.0)
        self.assertEqual(payload["trainer_args"]["ttc_max_range_m"], 20.0)
        self.assertEqual(
            observation_protocol_version(20.0),
            "carla_obs_19_lidar20m_v2",
        )

    def test_town05_v3_preserves_science_and_binds_new_c0_identity(self):
        v2 = load_experiment_contract(
            TOWN05_V2_CONTRACT_PATH,
            project_dir=PROJECT_DIR,
        )["payload"]
        v3 = load_experiment_contract(
            TOWN05_V3_CONTRACT_PATH,
            project_dir=PROJECT_DIR,
        )["payload"]
        self.assertEqual(v3["contract_id"], "P0_BASE_MAPPO_TOWN05_V3")
        self.assertEqual(v3["trainer_args"], v2["trainer_args"])
        self.assertEqual(v3["offroad_protocol"], v2["offroad_protocol"])
        self.assertEqual(
            v3["evaluation_contract"]["protocol_version"],
            "p0_base_c0_eval_town05_v3",
        )

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
    def test_monitor_options_preserve_frozen_scientific_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir) / "P0_BASE_MAPPO_Town02_s42_20260804_210000"
            baseline = _formal_args(save_dir)
            args = build_parser().parse_args([
                "--monitor-mode", "web", "--monitor-view", "overhead",
                "--monitor-overhead-height", "50", "--monitor-host", "127.0.0.1",
                "--monitor-port", "8081", "--monitor-fps", "2",
                "--monitor-width", "640", "--monitor-jpeg-quality", "50",
                "--monitor-record-dir", str(Path(temp_dir) / "frames"),
                "--monitor-smoothing-seconds", "0.7",
                "--monitor-service-dir", str(Path(temp_dir) / "service"),
            ], namespace=copy.deepcopy(baseline))
            args._explicit_cli_options.update({
                "--monitor-mode", "--monitor-view", "--monitor-overhead-height",
                "--monitor-host", "--monitor-port", "--monitor-fps",
                "--monitor-width", "--monitor-jpeg-quality", "--monitor-record-dir",
                "--monitor-smoothing-seconds", "--monitor-service-dir",
            })
            apply_experiment_contract(baseline, project_dir=PROJECT_DIR)
            apply_experiment_contract(args, project_dir=PROJECT_DIR)
            self.assertEqual(args.monitor_view, "overhead")
            self.assertEqual(args.monitor_overhead_height, 50)
            self.assertEqual(args.monitor_smoothing_seconds, 0.7)
            self.assertEqual(
                {name: getattr(args, name) for name in TRAINER_ARG_NAMES},
                {name: getattr(baseline, name) for name in TRAINER_ARG_NAMES},
            )

    def test_town05_contract_drives_both_ppo_and_environment_reward_gamma(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_id = "P0_BASE_MAPPO_Town05_s42_20260812_120000"
            argv = [
                "--experiment-contract",
                str(TOWN05_CONTRACT_PATH),
                "--updates",
                "5",
                "--run-id",
                run_id,
                "--save-dir",
                str(Path(temp_dir) / run_id),
            ]
            args = build_parser().parse_args(argv)
            args._explicit_cli_options = {
                token.split("=", 1)[0]
                for token in argv
                if token.startswith("--")
            }
            metadata = apply_experiment_contract(args, project_dir=PROJECT_DIR)
            kwargs = build_training_environment_kwargs(args, num_agents=2)

            self.assertEqual(metadata["payload"]["trainer_args"]["gamma"], 0.997)
            self.assertEqual(args.gamma, 0.997)
            self.assertEqual(kwargs["reward_gamma"], args.gamma)

            matching_env = SimpleNamespace(reward_gamma=kwargs["reward_gamma"])
            validate_environment_reward_discount(matching_env, args)
            with self.assertRaisesRegex(
                CheckpointProtocolError, "reward discount mismatch"
            ):
                validate_environment_reward_discount(
                    SimpleNamespace(reward_gamma=0.99), args
                )

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
            with (run_dir / RESOLVED_CONFIG_NAME).open(
                "r", encoding="utf-8"
            ) as handle:
                resolved = json.load(handle)
            self.assertEqual(
                set(resolved["resolved_training_config"]),
                set(TRAINER_ARG_NAMES),
            )
            self.assertEqual(resolved["resolved_training_config"]["seed"], 42)
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

    def test_resume_rejects_resolved_seed_drift(self):
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
            resolved_path = run_dir / RESOLVED_CONFIG_NAME
            with resolved_path.open("r", encoding="utf-8") as handle:
                resolved = json.load(handle)
            resolved["resolved_training_config"]["seed"] = 7
            with resolved_path.open("w", encoding="utf-8") as handle:
                json.dump(resolved, handle)
            with self.assertRaisesRegex(
                ExperimentContractError, "resolved training config mismatch"
            ):
                validate_resume_experiment_evidence(args=args, run_dir=run_dir)


if __name__ == "__main__":
    unittest.main()
