"""Trusted local checkpoint protocol; restart only at completed A/B round boundaries."""
import hashlib
import json
from pathlib import Path
import platform
import shutil
import importlib.metadata
import torch
import os
from ..config import ROOT

PROTOCOL = "highway_team_boundary_v2"
RUNTIME_PACKAGES = ("torch", "numpy", "gymnasium", "highway-env", "tensorboard", "pygame", "scipy", "pillow", "matplotlib")


def runtime_versions():
    return {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES}


def validate_runtime(recorded):
    if recorded != runtime_versions():
        raise ValueError("runtime dependency versions changed; restore the original environment or start a new run")


def source_files():
    """Prune large mutable artifact trees before walking; safe with concurrent runs."""
    files=[]
    for directory,dirs,names in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in {'runs','dist','__pycache__','.git'}]
        files.extend(Path(directory)/name for name in names if name.endswith('.py'))
    return sorted(files,key=lambda p:p.relative_to(ROOT).as_posix())


def source_digest():
    digest = hashlib.sha256()
    for path in source_files():
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode("utf-8"))
    return digest.hexdigest()


def provenance(output, config, initialization=None):
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    manifest = {"schema": PROTOCOL, "config_sha256": config.digest(), "source_sha256": source_digest(),
                "python": platform.python_version(), "platform": platform.platform(),
                "device": "cpu", "initialization": initialization,
                "packages": runtime_versions(),
                "budget": "A+B policy decisions; each decision advances two blue vehicles",
                "backend": config.backend,
                "red_controller": ("frozen external CARLA controller; see backend_contract" if config.backend == "carla" else
                    f"fixed IDM target_speed={config.task.red_target_speed if config.task.research_protocol != 'legacy_v2' else 18}, "
                    f"lane_change={config.task.red_lane_change if config.task.research_protocol != 'legacy_v2' else False}")}
    if config.task.research_protocol=='bottleneck_paired_attack_v4':
        manifest['red_controller']='one frozen IDM red0; case target speed=base +/-2m/s; explicit merge route'
        manifest['physical_roles']={'red':[0],'blue':[1,2]}
        manifest['gate_scope']='scenario-family quality, paired initial-state hashes; KL uses identical student input rows'
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for path in source_files():
        relative = path.relative_to(ROOT)
        if not {'runs','dist','__pycache__'}.intersection(relative.parts):
            target = output / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    shutil.copy2(ROOT / "requirements.txt", output / "source" / "requirements.txt")
    shutil.copy2(ROOT / "requirements-server.txt", output / "source" / "requirements-server.txt")


def save(path, state):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)


def read(path):
    # Only load trusted, locally produced checkpoints: optimizer/RNG states use pickle.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("protocol") != PROTOCOL:
        raise ValueError("not a Highwaytest round-boundary checkpoint; CARLA weights are incompatible")
    return state
