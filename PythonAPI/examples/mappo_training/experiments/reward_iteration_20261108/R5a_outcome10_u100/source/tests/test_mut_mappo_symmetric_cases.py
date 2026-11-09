"""Consequential symmetry/split checks; no CARLA Client or simulation."""
import copy
import json
import unittest
from collections import Counter

from utils.build_mut_mappo_symmetric_cases_v2 import (
    DEFAULT_OUTPUT, DIRECTIONS, FAMILIES, ROOT, approximate_initial_states,
    build_payloads, configuration_for_manifest, validate_manifest, write_payloads,
)


class SymmetricCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = build_payloads()
        cls.manifest = cls.files["cases.json"]
        prefix = DEFAULT_OUTPUT.relative_to(ROOT).as_posix()
        cls.bases = {f"{prefix}/{name}": value for name, value in cls.files.items() if name.startswith("bases/")}
        cls.lookup = {c["case_key"]: c for c in cls.manifest["cases"]}

    def test_complete_balanced_orbits_and_four_episode_collection_blocks(self):
        self.assertEqual({s: len(k) for s, k in self.manifest["splits"].items()},
                         {"train": 96, "gate": 32, "selection": 32, "test": 32})
        for keys in self.manifest["splits"].values():
            for start in range(0, len(keys), 4):
                rows = [self.lookup[key] for key in keys[start:start + 4]]
                self.assertEqual({c["direction"] for c in rows}, set(DIRECTIONS))
                self.assertEqual(Counter(c["family"] for c in rows), dict.fromkeys(FAMILIES, 2))
                self.assertEqual(sum(c["red0_left"] for c in rows), 2)
                self.assertEqual(sum(c["blue2_near"] for c in rows), 2)
        self.assertEqual(len(self.manifest["cases"]), 192)

    def test_blue_order_partners_exchange_whole_gap_and_speed_pairs(self):
        for case in self.manifest["cases"]:
            partners = [c for c in self.manifest["cases"] if all(c[k] == case[k] for k in
                        ("parameter_id", "direction", "family", "red0_left"))
                        and c["blue2_near"] != case["blue2_near"]]
            self.assertEqual(len(partners), 1)
            for own, other in (("agent2", "agent3"), ("agent3", "agent2")):
                for field in ("forward_m", "initial_speed_mps"):
                    self.assertEqual(case["blue_spawn_offsets"][own][field],
                                     partners[0]["blue_spawn_offsets"][other][field])

    def test_red_role_partners_exchange_actual_source_poses(self):
        prefix = DEFAULT_OUTPUT.relative_to(ROOT).as_posix()
        for direction in DIRECTIONS:
            left = self.bases[f"{prefix}/bases/{direction}_red0_left.json"]
            right = self.bases[f"{prefix}/bases/{direction}_red0_right.json"]
            self.assertEqual(left["agents"]["agent0"], right["agents"]["agent1"])
            self.assertEqual(left["agents"]["agent1"], right["agents"]["agent0"])
            self.assertEqual(left["goal_zone"], right["goal_zone"])

    def test_no_seed_or_label_only_copies_and_no_cross_split_parameter_orbits(self):
        signatures, membership = set(), {}
        for split, keys in self.manifest["splits"].items():
            for key in keys:
                case = self.lookup[key]
                self.assertEqual(membership.setdefault(case["parameter_id"], split), split)
                agents = approximate_initial_states(case, self.bases[case["base_scenario"]])
                signature = json.dumps(agents, sort_keys=True)
                self.assertNotIn(signature, signatures)
                signatures.add(signature)
        # Re-keying/re-seeding and relabeling do not evade physical de-duplication.
        changed = copy.deepcopy(self.manifest)
        duplicate = copy.deepcopy(changed["cases"][0])
        duplicate.update(case_key="renamed_copy", seed=999999, family="cut_in")
        changed["cases"].append(duplicate)
        changed["splits"]["train"].append(duplicate["case_key"])
        with self.assertRaisesRegex(ValueError, "physical initial state"):
            validate_manifest(changed, self.bases)

    def test_missing_partner_or_moving_orbit_member_between_splits_is_rejected(self):
        changed = copy.deepcopy(self.manifest)
        key = changed["splits"]["train"].pop()
        changed["cases"] = [c for c in changed["cases"] if c["case_key"] != key]
        with self.assertRaisesRegex(ValueError, "symmetry orbit"):
            validate_manifest(changed, self.bases)
        changed = copy.deepcopy(self.manifest)
        changed["splits"]["test"].append(changed["splits"]["train"].pop())
        with self.assertRaisesRegex(ValueError, "symmetry orbit"):
            validate_manifest(changed, self.bases)

    def test_generated_files_and_legacy_contract_compatibility(self):
        write_payloads(self.files, check=True)
        from training.mut_mappo_config import load_config, validate_config, cases_for
        original = load_config("configs/mut_mappo/local_candidate_v1.json")
        source = {k: v for k, v in original.items() if not k.startswith("_")}
        candidate = configuration_for_manifest(source, DEFAULT_OUTPUT / "cases.json")
        self.assertEqual({k: v for k, v in candidate.items() if k != "case_manifest"},
                         {k: v for k, v in source.items() if k != "case_manifest"})
        checked = validate_config(candidate)
        self.assertEqual(len(cases_for(checked, "train")), 96)
        self.assertEqual(checked["_root_validation"]["split_counts"]["gate"], 32)
        self.assertEqual(original["case_manifest"], "data/mut_mappo/cases_v1.json")
        self.assertEqual(len(checked["_root_validation"]["scenario_hashes"]), 8)


if __name__ == "__main__":
    unittest.main()
