"""Prepare small local readiness configurations; never start CARLA or training."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
from training.checkpoint_protocol import atomic_write_json
from training.mut_mappo_config import ROOT, load_config, portable_config, validate_config


def prepare(output):
    output = Path(output)
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    if output.exists():
        raise FileExistsError("readiness configurations exist; preserve them and use a new directory")
    base = portable_config(load_config("configs/mut_mappo/local_candidate_v1.json"))
    variants = {}
    for name, mode in (("b0", "off"), ("b1", "plain"), ("b2", "selective")):
        cfg = copy.deepcopy(base)
        cfg.update(method=mode, algorithm="mappo", team_count=2, research_status="candidate")
        cfg["run"].update(rounds=2, train_tick_budget=12000, stop_rule="rounds", schedule_clock="rounds")
        cfg["mutual"].update(warmup_rounds=1 if mode != "off" else 5,
                             refresh_rounds=10, schedule_clock="rounds")
        # Short warmup is deliberately an engineering intervention.  These
        # trajectories cannot be relabeled as the standard pilot experiment.
        variants[name] = cfg
    variants["red_u456"] = copy.deepcopy(base)
    variants["red_u356"] = copy.deepcopy(base)
    variants["red_u356"]["red_checkpoint"].update(
        path="logs/P0_BASE_MAPPO_Town05R8C1PaceU100_s42_20260903_180224/latest.pt",
        sha256="eea971a82ef6293357161fad40bcf58fa91663a9282c60dedbcfc78193dfe371")
    checks = {name: validate_config(cfg)["_root_validation"] for name, cfg in variants.items()}
    output.mkdir(parents=True)
    for name, cfg in variants.items():
        atomic_write_json(cfg, output / (name + ".json"))
    record = {"schema_version": "mut_mappo_local_readiness_v1", "status": "PREPARED_NOT_RUN",
              "purpose": "engineering_only_not_pilot_or_final_test", "configurations": checks,
              "sequence": ["red_u456_C0_C1", "optional_red_u356_C0_C1", "b0_round1_resume2",
                           "b0_independent_selection", "b1_round1_resume2", "b2_round1_resume2"],
              "max_concurrent_tick_owners": 1, "user_started_carla_required": True,
              "new_runs_written": False}
    atomic_write_json(record, output / "readiness_manifest.json")
    return {"status": record["status"], "directory": str(output), "configurations": list(variants),
            "carla_connected": False, "training_launched": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(prepare(parser.parse_args().output), ensure_ascii=False, indent=2))
