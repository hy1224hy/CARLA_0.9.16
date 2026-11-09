import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from training.base_driving_contract import configure, load_contract, validate_resume_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/base_driving/base_driving_v6_v1.json"


def arguments():
    defaults = json.loads(CONTRACT.read_text(encoding="utf-8"))["training_overrides"]
    return SimpleNamespace(**defaults, base_driving_contract=str(CONTRACT),
                           experiment_contract="", continuation_contract="",
                           allow_resume_code_drift=False, scenario_dir="",
                           _explicit_cli_options=set())


class ContractTests(unittest.TestCase):
    def test_configuration_is_bound_to_pool_qualification_and_unchanged_20_2_parent(self):
        args = arguments()
        metadata = configure(args, ROOT)
        self.assertEqual(args.reward_version, "v6")
        self.assertEqual(args.pace_profile, "instant_pace_v6")
        self.assertEqual(args.rollout_episodes, 16)
        self.assertEqual(args.scenario_sampling_mode, "base_driving_paired_v1")
        for field in ("contract_sha256", "qualification_contract_sha256", "scenario_pool_sha256"):
            self.assertEqual(len(metadata["record"][field]), 64)

    def test_strict_resume_rejects_legacy_reward_and_contract_drift(self):
        args = arguments()
        record = configure(args, ROOT)["record"]
        validate_resume_contract(args, {"base_driving_contract": record})
        with self.assertRaises(ValueError):
            validate_resume_contract(args, {})
        changed = copy.deepcopy(record)
        changed["reward_profile"] = "startup_v1"
        with self.assertRaises(ValueError):
            validate_resume_contract(args, {"base_driving_contract": changed})
        with self.assertRaises(ValueError):
            validate_resume_contract(SimpleNamespace(), {"base_driving_contract": record})
        with self.assertRaises(ValueError):
            validate_resume_contract(SimpleNamespace(), {"protocol_versions": {"base_driving": "base_driving_vnext_v1"}})

    def test_no_explicit_hyperparameter_drift_or_code_hash_bypass(self):
        args = arguments()
        args.gamma = 0.99
        args._explicit_cli_options = {"--gamma"}
        with self.assertRaises(ValueError):
            configure(args, ROOT)
        args = arguments()
        args.allow_resume_code_drift = True
        with self.assertRaises(ValueError):
            configure(args, ROOT)

    def test_reward_ablation_changes_only_profile_and_contract_identity(self):
        v6 = load_contract(CONTRACT, ROOT)
        startup = load_contract(ROOT / "configs/base_driving/base_driving_startup_v1_v1.json", ROOT)
        left, right = copy.deepcopy(v6["payload"]), copy.deepcopy(startup["payload"])
        left.pop("reward_profile")
        right.pop("reward_profile")
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
