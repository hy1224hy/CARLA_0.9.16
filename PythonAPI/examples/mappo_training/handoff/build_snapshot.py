"""Build a local, explicit handoff snapshot; no branch or simulator mutations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[2]
OUT = Path(__file__).resolve().parent
HANDOFF_DATE = "2026-10-09"
BASE_SHA = "f7be58a9881baac14ad63c95cd94712ab8343f35"

DOC_NAMES = [
    "开发交接指令_MutMAPPO_20261009.md",
    "最快获得实验结果_执行顺序_20261009.md",
    "相关论文检索与实施细节依据_20261008.md",
    "MutMAPPO实施计划完整性审查与待定事项_20261008.md",
    "MutMAPPO最小方法与对照实验规格_20261007.md",
    "MutDQN原文解读与第一研究点迁移建议_20261007.md",
    "两研究点修订_互学习通信与驾驶风格种群_20261007.md",
    "中期材料与两项研究实验计划_20261007.md",
    "毕业课题_研究推进TODO.md",
    "论文研究材料与方案讨论_持续更新.md",
    "R8C1_u456_训练结果分析与改进方向_20260928.md",
    "R8C1_u356_新奖励微调启动记录_20260927.md",
    "基础策略代码审查与推进路线_20260927.md",
    "P0-R8C1_长训练续训与监控计划_20260915.md",
    "R8C1_u208_行为诊断结果_20260919.md",
    "R8C1_路线进度奖励调整_20260925.md",
]
RUNS = {
    "u356": (
        "P0_BASE_MAPPO_Town05R8C1PaceU100_s42_20260903_180224",
        "eea971a82ef6293357161fad40bcf58fa91663a9282c60dedbcfc78193dfe371",
    ),
    "u456": (
        "R8C1_RoutePaceV1_u356_to406_s42_20260927",
        "8b9cb173cd70aacc201f0cb68183fdfeeb593cc8f29863c096a5f59076954545",
    ),
}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.quotepath=false", *args], cwd=REPO,
        check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()


def main() -> None:
    actual_sha = git("rev-parse", "HEAD")
    if actual_sha != BASE_SHA:
        raise RuntimeError("Source HEAD changed; update the handoff before rebuilding.")
    sources: list[tuple[Path, str]] = [(REPO / "AGENTS.md", "repo/AGENTS.md")]
    for name in DOC_NAMES:
        source = PROJECT / "docs" / name
        sources.append((source, "repo/" + source.relative_to(REPO).as_posix()))
    for label, (run_name, expected) in RUNS.items():
        run = PROJECT / "logs" / run_name
        model = run / "latest.pt"
        if hashlib.sha256(model.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Candidate checkpoint changed: {label}")
        for name in ("latest.pt", "latest.pt.sha256", "checkpoint_manifest.json"):
            sources.append((run / name, f"assets/{label}/{name}"))
    new_run = PROJECT / "logs" / RUNS["u456"][0]
    for name in ("reward_fork.json", "metrics_schema.json"):
        sources.append((new_run / name, f"assets/u456/{name}"))
    for name in ("comparison_u456.json", "speed_safety_u456.json", "completion_u456.json"):
        sources.append((new_run / "session" / "monitoring" / name, f"assets/u456/evidence/{name}"))

    entries = []
    payloads = []
    for source, archive_path in sources:
        data = source.read_bytes()
        entries.append({
            "source": str(source.resolve()), "archive_path": archive_path,
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        })
        payloads.append((archive_path, data))
    manifest = {
        "handoff_date": HANDOFF_DATE, "timezone": "Asia/Shanghai",
        "source_repository": str(REPO), "source_branch": git("branch", "--show-current"),
        "source_head": actual_sha,
        "tracked_worktree_changes": git("diff", "--name-only").splitlines(),
        "archive_scope": "Selected documents, candidate weights and limited evidence only",
        "excluded": ["source tree", "CARLA engine", "venv", "full historical logs", "periodic checkpoints", "user PDF"],
        "files": entries,
    }
    readme = (
        "Mut-MAPPO handoff snapshot, 2026-10-09 Asia/Shanghai\n"
        "Read repo/PythonAPI/examples/mappo_training/docs/开发交接指令_MutMAPPO_20261009.md first.\n"
        f"Start source code from Git HEAD {BASE_SHA}.\n"
        "Uncommitted documents are included as complete snapshots, not automatically inherited by worktrees.\n"
        "Compare and merge repo/ documents into the target checkout; preserve newer user edits.\n"
        "assets/ weights are candidates for screening or explicit initialization, not proof of a qualified policy.\n"
        "This archive lacks full old-run logs and does not enable strict historical training resume.\n"
        "Use the original run directories read-only for historical recovery/evidence.\n"
        "Neither this archive nor its instructions creates a branch, approves pending research choices, or starts CARLA.\n"
        "All local worktrees share CARLA port 2000; only one process may advance its clock.\n"
    )
    archive = OUT / "MutMAPPO_20261009.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for archive_path, data in payloads:
            bundle.writestr(archive_path, data)
        bundle.writestr("HANDOFF_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        bundle.writestr("README.txt", readme)
    with ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Archive CRC verification failed.")
        for entry in entries:
            data = bundle.read(entry["archive_path"])
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise RuntimeError("Archive payload hash verification failed.")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (OUT / "HANDOFF_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.txt").write_text(readme, encoding="utf-8")
    (OUT / "MutMAPPO_20261009.zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    print(json.dumps({"archive": str(archive), "files": len(entries), "bytes": archive.stat().st_size, "sha256": digest, "verified": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
