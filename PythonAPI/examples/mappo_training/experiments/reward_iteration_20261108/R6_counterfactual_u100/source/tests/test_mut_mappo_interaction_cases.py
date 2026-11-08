"""Consequential invariants for the small sequential/crossing scene pool."""
import copy
import importlib
import json
from collections import Counter, defaultdict
import unittest
from unittest.mock import patch

from training.first_point_scenarios import ROOT, sample_pairs


class InteractionCaseTests(unittest.TestCase):
    def setUp(self):
        self.client_guard = patch("carla.Client", side_effect=AssertionError("offline scene tests cannot connect"))
        self.client_guard.start()
        self.addCleanup(self.client_guard.stop)
        try:
            self.builder = importlib.import_module("utils.build_mut_mappo_interaction_cases_v3")
        except ModuleNotFoundError:
            self.fail("small sequential/crossing scene generator is not implemented")
        self.files = self.builder.build_payloads()
        self.manifest = self.files["cases.json"]
        self.bases = {str(self.builder.DEFAULT_OUTPUT.relative_to(ROOT) / name).replace("\\", "/"): payload
                      for name, payload in self.files.items() if name.startswith("bases/")}

    def test_small_pool_has_complete_mirror_and_joint_slot_orbits(self):
        lookup = {c["case_key"]: c for c in self.manifest["cases"]}
        for split, keys in self.manifest["splits"].items():
            cases = [lookup[k] for k in keys]
            self.assertEqual(len(cases), 24)
            self.assertEqual(Counter(c["family"] for c in cases), {"blocking": 8, "cut_in": 16})
            for direction in self.builder.DIRECTIONS:
                own = [c for c in cases if c["direction"] == direction]
                self.assertEqual({(c["layout"], c["front_side"], c["red0_left"]) for c in own},
                    {("sequential", "both", side) for side in (False, True)} |
                    {("crossing", front, side) for front in ("left", "right") for side in (False, True)})
                pairs = defaultdict(list)
                for c in own:
                    pairs[c["pair_id"]].append(c)
                self.assertEqual(sorted(map(len, pairs.values())), [2, 2, 2])
                # Joint red+blue slot exchange keeps the physical geometry,
                # and gives each actor both the front/rear role in crossing.
                for pair in pairs.values():
                    self.assertEqual({c["red0_left"] for c in pair}, {False, True})
                cross = [c for c in own if c["layout"] == "crossing"]
                for aid in (2, 3):
                    signs = [c["blue_spawn_offsets"][f"agent{aid}"]["forward_m"] > 0 for c in cross]
                    self.assertEqual(sum(signs), 2)

    def test_removing_mirror_or_copying_a_physical_split_fails(self):
        bad = copy.deepcopy(self.manifest)
        victim = bad["splits"]["train"].pop()
        bad["cases"] = [c for c in bad["cases"] if c["case_key"] != victim]
        with self.assertRaisesRegex(ValueError, "symmetry|orbit"):
            self.builder.validate_manifest(bad, self.bases)
        bad = copy.deepcopy(self.manifest)
        first = next(c for c in bad["cases"] if c["case_key"] == bad["splits"]["train"][0])
        target = next(c for c in bad["cases"] if c["case_key"] == bad["splits"]["gate"][0])
        target.update(copy.deepcopy({k: v for k, v in first.items() if k not in ("case_key", "split", "seed", "pair_id")}))
        with self.assertRaisesRegex(ValueError, "physical|geometry"):
            self.builder.validate_manifest(bad, self.bases)

    def test_rear_speed_and_protection_separation_have_nominal_interaction_room(self):
        for case in self.manifest["cases"]:
            for item in case["blue_spawn_offsets"].values():
                gap, speed = item["forward_m"], item["initial_speed_mps"]
                self.assertLessEqual(speed, 15 / 3.6)
                if gap < 0:
                    closing = speed - 2.5
                    self.assertGreater(closing, 0)
                    self.assertGreaterEqual(abs(gap) - closing * 3., 6.)
                    self.assertLess((abs(gap)-6.)/closing, (32.-3.)/2.5)
                else:
                    self.assertGreater((32.-gap-3.) / speed - 3., 3.)

    def test_existing_pair_sampler_covers_all_templates_without_splitting_roles(self):
        cases = [c for c in self.manifest["cases"] if c["split"] == "train"]
        seen = set()
        for update in range(8):
            sampled = sample_pairs(cases, update, 4)
            self.assertEqual(Counter(c["family"] for c in sampled), {"blocking": 2, "cut_in": 2})
            self.assertEqual(sampled[0]["pair_id"], sampled[1]["pair_id"])
            self.assertEqual(sampled[2]["pair_id"], sampled[3]["pair_id"])
            seen.update(c["case_key"] for c in sampled)
        self.assertEqual(len(seen), 24)

    def test_production_resolver_checks_every_spawn_goal_and_crossing_boundary(self):
        report = self.builder.validate_opendrive(self.manifest, self.bases)
        self.assertEqual(report["cases_checked"], 96)
        self.assertEqual(report["world_ticks"], 0)
        self.assertFalse(report["carla_client_created"])
        self.assertEqual(report["crossing_cases_with_legal_adjacent_lane"], 64)

    def test_new_candidate_keeps_red_reward_budget_and_uses_new_reference_identity(self):
        source = json.loads((ROOT / "experiments/adversarial_u1000_pilot_20261028_v1/base_config.json").read_text())
        new = self.builder.candidate_config(source)
        self.assertEqual(source["red_checkpoint"], new["red_checkpoint"])
        self.assertEqual(source["learning"], new["learning"])
        self.assertEqual(source["task"]["reward"], new["task"]["reward"])
        self.assertEqual(source["run"], new["run"])
        self.assertEqual(new["run"]["train_tick_budget"], 24000)
        self.assertNotEqual(source["case_manifest"], new["case_manifest"])
        self.assertNotEqual(source["task"]["team_reward"]["reference_dir"], new["task"]["team_reward"]["reference_dir"])


if __name__ == "__main__":
    unittest.main()
