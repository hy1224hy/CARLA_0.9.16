"""Server preflight, matrix fairness checks and bounded subprocess scheduling."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from .config import ROOT, load
from .training.checkpoint import source_digest


def comparison_contract(config, initialization=None):
    data = config.to_dict()
    data.pop("group")
    data.pop("seed")
    data["initialization_mode"] = "normal_driving" if initialization else "scratch"
    data["pretraining_steps"] = initialization["prior_train_steps"] if initialization else 0
    return data


def preflight(config_dir):
    if sys.version_info < (3, 10):
        raise RuntimeError("Python >=3.10 required")
    paths = sorted(Path(config_dir).glob("*.json"))
    if not paths:
        raise ValueError(f"no configs: {config_dir}")
    configs = [load(path) for path in paths]
    if any(c.backend != "highway" for c in configs):
        raise ValueError("CARLA studies require the injected train_carla scene-owner entry, not Highway batch")
    identities = [(c.group, c.seed) for c in configs]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate (group,seed) in matrix")
    first = comparison_contract(configs[0])
    if any(comparison_contract(c) != first for c in configs):
        raise ValueError("matrix mixes tasks, budgets or learning settings; use separate matrix directories")
    required = ("torch", "numpy", "gymnasium", "highway-env", "tensorboard", "pygame", "scipy", "pillow", "matplotlib")
    versions = {name: importlib.metadata.version(name) for name in required}
    if versions["highway-env"] != "1.10.2" or versions["gymnasium"] != "1.1.1":
        raise ValueError("install the pinned Highway-env/Gymnasium versions")
    import torch
    from .environment.factory import make_env
    import numpy as np
    env = make_env(configs[0].task, configs[0].gamma)
    obs, info = env.reset(seed=0)
    env.step(np.zeros((2, 2), np.float32))
    frame = env.render()
    env.close()
    return {"status": "passed", "python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "source_sha256": source_digest(), "runs": len(paths),
            "groups": sorted({c.group for c in configs}), "seeds": sorted({c.seed for c in configs}),
            "training_device": "cpu", "cuda_available_but_unused": torch.cuda.is_available(),
            "cpu_count": os.cpu_count(), "threads_per_run": configs[0].threads,
            "total_train_steps": sum(c.total_train_steps for c in configs),
            "obs_shape": list(obs.shape), "state_shape": list(info["state"].shape),
            "headless_frame_shape": list(frame.shape), "contract": first}


def batch(config_dir, output, jobs=1, initialize_from=None):
    from .training.locking import run_lock
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("batch output must be under highwaytest/runs")
    with run_lock(output / "batch_controller"):
        return _batch(config_dir, output, jobs, initialize_from)


def _batch(config_dir, output, jobs, initialize_from=None):
    import hashlib
    initial_hash = hashlib.sha256(Path(initialize_from).read_bytes()).hexdigest() if initialize_from else None
    if type(jobs) is not int or jobs < 1:
        raise ValueError("jobs must be positive")
    check = preflight(config_dir)
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("batch output must be under highwaytest/runs")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "batch_manifest.json"
    paths = sorted(Path(config_dir).glob("*.json"))
    matrix = {f"{c.group}_s{c.seed}": {"path": str(p.resolve()), "config_sha256": c.digest()}
              for p in paths for c in [load(p)]}
    identity = {name: item["config_sha256"] for name, item in matrix.items()}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get('initialization_sha256') != initial_hash:
            raise ValueError('batch initialization changed')
        if previous["identity"] != identity or previous["preflight"]["source_sha256"] != check["source_sha256"]:
            raise ValueError("existing batch has different configuration or source")
        if previous["preflight"]["packages"] != check["packages"]:
            raise ValueError("batch runtime dependency versions changed")
    else:
        manifest_path.write_text(json.dumps({"identity": identity, "matrix": matrix, "preflight": check,
            'initialization_sha256':initial_hash}, indent=2), encoding="utf-8")

    def worker(name, item):
        run_dir = output / name
        summary = run_dir / "summary.json"
        if summary.exists() and json.loads(summary.read_text())["status"] == "complete":
            return {"name": name, "status": "skipped_complete"}
        checkpoint = run_dir / "latest.pt"
        if run_dir.exists() and not checkpoint.exists():
            return {"name": name, "status": "needs_attention", "reason": "existing run has no committed checkpoint; use a fresh batch directory"}
        command = [sys.executable, "-u", "-m", "highwaytest", "train"]
        command += ["--resume", str(checkpoint)] if checkpoint.exists() else [
            "--config", item["path"], "--output", str(run_dir)]
        if initialize_from and not checkpoint.exists():
            command += ['--initialize-from',str(Path(initialize_from).resolve())]
        started = time.time()
        with (output / f"{name}.console.log").open("a", encoding="utf-8") as stream:
            stream.write(f"\nSTART {json.dumps(command)}\n")
            stream.flush()
            result = subprocess.run(command, cwd=ROOT.parent, stdout=stream, stderr=subprocess.STDOUT,
                                    env=os.environ | {"PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg",
                                                      "OMP_NUM_THREADS": str(check["threads_per_run"]),
                                                      "MKL_NUM_THREADS": str(check["threads_per_run"])})
        return {"name": name, "status": "complete" if result.returncode == 0 else "failed",
                "exit_code": result.returncode, "elapsed_s": time.time() - started}

    results = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        pending = [pool.submit(worker, name, item) for name, item in matrix.items()]
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
            temp = output / "batch_status.tmp"
            temp.write_text(json.dumps(results, indent=2), encoding="utf-8")
            temp.replace(output / "batch_status.json")
    if any(r["status"] in ("failed", "needs_attention") for r in results):
        raise RuntimeError("some runs did not complete; inspect batch_status.json and console logs")
    return results
