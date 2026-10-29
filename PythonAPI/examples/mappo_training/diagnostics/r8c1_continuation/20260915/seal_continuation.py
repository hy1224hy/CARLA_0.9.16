"""One-time freeze of the user-approved u600 budget extension and u100 evidence."""
import json
from pathlib import Path
import shutil

from training.checkpoint_protocol import load_checkpoint_file, sha256_file, atomic_write_json

root = Path.cwd()
evidence = root / "diagnostics/r8c1_continuation/20260915"
run_id = "P0_BASE_MAPPO_Town05R8C1PaceU100_s42_20260903_180224"
run = root / "logs" / run_id
parent_sources = json.loads((evidence / "parent_sources.json").read_text())
checkpoint, digest = load_checkpoint_file(run / "latest.pt", map_location="cpu")
assert checkpoint["update"] == 100 and checkpoint["total_episodes"] == 1600
assert digest == "1e975bff5975060faeeaedbff4fa42a7f94aebc4bd694b1a6d4fe7e2ad815904"
backup = evidence / "parent_u100"
backup.mkdir(exist_ok=True)
for name in ("latest.pt", "latest.pt.sha256", "checkpoint_manifest.json", "experiment_manifest.json",
             "resolved_config.json", "metrics.jsonl", "episodes.jsonl"):
    dst = backup/name
    if not dst.exists():
        shutil.copy2(run/name,dst)
    assert sha256_file(dst) == sha256_file(run/name)
names = set(parent_sources) | {"training/continuation_protocol.py", "utils/r8c1_training_session.py", "resume_r8c1_u600.ps1"}
pins = {name:sha256_file(root/name) for name in sorted(names)}
changed = sorted(name for name in parent_sources if pins[name] != parent_sources[name])
assert changed == ["training/experiment_contract.py", "training/train_no_rllib_mappo_robust_v2.py"]
payload = {"version":"r8c1_budget_continuation_v1", "id":"R8C1_U100_TO_U600_20260915_V1",
    "run_id":run_id,"parent_update":100,"planned_total_updates":600,
    "parent_contract_sha256":checkpoint["run_metadata"]["experiment_contract_sha256"],
    "parent_checkpoint_sha256":digest,
    "parent_code_fingerprint":checkpoint["run_metadata"]["code_fingerprint"],
    "parent_git_sha":checkpoint["run_metadata"]["last_git_sha"],
    "parent_source_sha256":parent_sources,"source_sha256":pins,"allowed_changed_sources":changed,
    "authorization":"User confirmed the existing 12-scenario, full-state u100-to-u600 implementation plan.",
    "scientific_changes":[],"formal_performance_eligibility":False}
target=root/"configs/base_mappo/p0_r8c1_continue_u600_v1.json"
if target.exists():
    assert json.loads(target.read_text()) == payload, "sealed extension must not be silently changed"
else:
    atomic_write_json(payload,target)
print(json.dumps({"extension":str(target),"sha256":sha256_file(target),"pinned_files":len(pins),"backup":str(backup)}))
