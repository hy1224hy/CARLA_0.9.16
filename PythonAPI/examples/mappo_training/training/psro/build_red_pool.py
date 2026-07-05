"""
Phase 0 辅助脚本：组建 RED 池
=================================
RED 池本质上就是"一个目录，里面放若干个兼容的 .pt"——RedTeamPool.from_dir 会加载目录下
所有 *.pt，只要求每个 ckpt 的 "actors" 里含 cooperative_ids 的权重。本脚本提供两种组池方式：

  ① 重训多 seed（方法 A，最贵但 RED 多样性最高）：
       python -m training.psro.build_red_pool --seeds 42 123 2024 --updates 150 \
           --out logs/red_pool/ckpts

  ② 动态收编已有策略（方法 B/C，**不训练**，秒级完成；从已训好的 ckpt 直接组池）：
       # 收多个 run 的 best.pt
       python -m training.psro.build_red_pool --from-ckpts \
           logs/exp/A0617_baseline_seed42/best.pt \
           logs/exp/A0610_baseline_seed42/best.pt \
           logs/exp/A0608_baseline_seed42/best.pt \
           --out logs/red_pool/ckpts
       # 或用 glob 收同一 run 的多个快照（不同 update = 不同策略）
       python -m training.psro.build_red_pool --from-ckpts \
           "logs/exp/A0617_baseline_seed42/mappo_robust_v2_update_1*00.pt" \
           --out logs/red_pool/ckpts

两种方式可叠加多次调用同一 --out 逐步扩池（默认不覆盖已存在的同名文件，除非 --overwrite）。

完成后：
    python -m training.psro.psro_runner --red-pool-dir logs/red_pool/ckpts ...

注意：
  - 方式①每个 seed 都会连 CARLA 跑满 --updates，请确保模拟器已启动；--collect-only 跳过训练
    只从已有 seed_<N>/ 目录收 best.pt。
  - 方式②纯文件操作 + 格式校验，不需要 CARLA。
  - 多样性提醒：若收编的 ckpt 都来自同一 seed（仅 update/日期不同），RED 行为差异有限，
    PSRO 的种群效应会偏弱；追求最大多样性仍建议混入不同 seed 重训的策略。
  - 收编后建议各 RED 单独评估 coop_success>80% 再入池（需 CARLA），见设计文档 Phase 0 验收。
"""
import argparse
import glob as _glob
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.append(str(_pkg_root))


def _seed_dir(base_logdir: str, seed: int) -> str:
    return os.path.join(base_logdir, f"seed_{seed}")


def train_one_seed(seed: int, updates: int, base_logdir: str, extra_args):
    """调用现有 robust_v2 训练脚本训一个 seed。失败抛 CalledProcessError。"""
    save_dir = _seed_dir(base_logdir, seed)
    cmd = [
        sys.executable, "-m", "training.train_no_rllib_mappo_robust_v2",
        "--seed", str(seed),
        "--updates", str(updates),
        "--save-dir", save_dir,
    ] + list(extra_args)
    print(f"\n{'='*70}\n[build_red_pool] 训练 seed={seed} → {save_dir}\n  cmd: {' '.join(cmd)}\n{'='*70}")
    subprocess.run(cmd, cwd=str(_pkg_root), check=True)
    return save_dir


def collect_best(seed: int, base_logdir: str, out_dir: str) -> str:
    """把 seed 训练目录下的 best.pt 拷到统一 RED 池目录，返回目标路径。"""
    src = os.path.join(_seed_dir(base_logdir, seed), "best.pt")
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"seed {seed} 未找到 best.pt: {src}\n"
            f"（训练可能未达到任何 best 窗口，或 --save-dir 不一致）"
        )
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f"red_seed{seed}.pt")
    shutil.copy2(src, dst)
    print(f"[build_red_pool] 收池: {src} -> {dst}")
    return dst


def _validate_ckpt(path: str, cooperative_ids: List[int]) -> str:
    """校验一个 ckpt 是否可作为 RED 入池：含 actors[cid]（cid ∈ cooperative_ids）。

    通过返回 ""；不通过返回原因字符串。延迟 import torch，避免方式②也要等 torch 初始化。
    """
    import torch
    try:
        ck = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        return f"无法加载: {e}"
    if not isinstance(ck, dict) or "actors" not in ck:
        return "缺少 'actors' 字段（不是合作策略 ckpt？）"
    actors = ck["actors"]
    missing = [cid for cid in cooperative_ids if cid not in actors]
    if missing:
        return f"actors 缺少 cooperative id {missing}（现有 {list(actors.keys())}）"
    return ""


def _expand_ckpt_args(patterns: List[str]) -> List[str]:
    """把 --from-ckpts 的参数展开成文件列表：支持直接路径、glob 通配、以及目录。

    目录 → 取其中所有 *.pt；glob → 展开；去重并保持顺序。
    """
    out: List[str] = []
    seen = set()
    for pat in patterns:
        if os.path.isdir(pat):
            hits = sorted(_glob.glob(os.path.join(pat, "*.pt")))
        elif any(ch in pat for ch in "*?[]"):
            hits = sorted(_glob.glob(pat))
        else:
            hits = [pat]
        for h in hits:
            ap = os.path.abspath(h)
            if ap not in seen:
                seen.add(ap)
                out.append(h)
    return out


def _pool_name_for(src: str) -> str:
    """给入池文件起一个唯一、可读的名字：<父目录>__<文件名>。"""
    src = os.path.normpath(src)
    parent = os.path.basename(os.path.dirname(src)) or "root"
    stem = os.path.basename(src)
    name = f"{parent}__{stem}"
    # 清洗非法字符
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)


def register_existing(patterns: List[str], out_dir: str, cooperative_ids: List[int],
                      link: bool = False, overwrite: bool = False):
    """方式②：把已有 ckpt 校验后收编进 RED 池目录（不训练）。返回 (收编列表, 失败列表)。"""
    files = _expand_ckpt_args(patterns)
    if not files:
        print(f"[build_red_pool] [!] --from-ckpts 没匹配到任何文件: {patterns}")
        return [], []
    os.makedirs(out_dir, exist_ok=True)
    collected, failures = [], []
    for src in files:
        if not os.path.exists(src):
            failures.append((src, "文件不存在")); print(f"  [X]   {src}: 文件不存在"); continue
        reason = _validate_ckpt(src, cooperative_ids)
        if reason:
            failures.append((src, reason)); print(f"  [X]   {src}: {reason}"); continue
        dst = os.path.join(out_dir, _pool_name_for(src))
        if os.path.exists(dst) and not overwrite:
            print(f"  [skip] 已存在: {os.path.basename(dst)}（--overwrite 可覆盖）")
            collected.append(dst); continue
        if os.path.exists(dst) and overwrite:
            os.remove(dst)
        try:
            if link:
                os.symlink(os.path.abspath(src), dst)
            else:
                shutil.copy2(src, dst)
        except OSError as e:
            # Windows 上 symlink 可能需要管理员权限，回退到复制
            if link:
                print(f"  (symlink 失败，回退复制: {e})")
                shutil.copy2(src, dst)
            else:
                failures.append((src, str(e))); print(f"  [X]   {src}: {e}"); continue
        print(f"  [OK]  {src}\n           -> {os.path.basename(dst)}")
        collected.append(dst)
    return collected, failures


def _print_summary(out_dir: str, collected, failures):
    print(f"\n{'='*70}\n[build_red_pool] 完成。RED 池目录: {os.path.abspath(out_dir)}")
    # 池里实际有多少个 .pt（含历史累积），而不仅本次收编
    total = len(_glob.glob(os.path.join(out_dir, "*.pt")))
    print(f"  本次入池 {len(collected)} 个；池内合计 {total} 个 .pt")
    if failures:
        print(f"  失败 {len(failures)} 个:")
        for s, r in failures:
            print(f"    - {s}: {r}")
    if total < 3:
        print("  [!] RED 池 < 3，PSRO 可能退化为 InRL。建议补足策略。")
    else:
        print("  [OK] 可执行下一步：")
        print(f"     python -m training.psro.psro_runner --red-pool-dir {out_dir} \\")
        print(f"         --scenario scenario_01_basic.json --psro-iters 5 --meta-solver prd")


def main():
    p = argparse.ArgumentParser(description="Phase 0：组建 RED 池（多 seed 重训 或 收编已有策略）")
    # 方式②：动态收编已有 ckpt（不训练）
    p.add_argument("--from-ckpts", type=str, nargs="+", default=None,
                   help="已有 ckpt 路径 / glob / 目录列表，校验后收编进 --out（不训练，不需 CARLA）")
    p.add_argument("--cooperative-ids", type=str, default="0,1",
                   help="RED 合作 id，用于校验 ckpt.actors 是否齐全")
    p.add_argument("--link", action="store_true",
                   help="收编时用符号链接代替复制（省空间；Windows 可能需管理员，失败自动回退复制）")
    p.add_argument("--overwrite", action="store_true",
                   help="收编时覆盖 --out 下的同名文件")
    # 方式①：重训多 seed
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024],
                   help="要重训的 seed 列表（≥3 个以避免 PSRO 退化为 InRL）")
    p.add_argument("--updates", type=int, default=150,
                   help="每个 seed 的训练 update 数（传给 robust_v2）")
    p.add_argument("--base-logdir", type=str, default="logs/red_pool",
                   help="各 seed 训练日志/ckpt 的父目录")
    p.add_argument("--collect-only", action="store_true",
                   help="跳过训练，仅从已有 seed_<N>/ 目录收集 best.pt")
    p.add_argument("--out", type=str, default="logs/red_pool/ckpts",
                   help="统一 RED 池目录（psro_runner --red-pool-dir 用它）")
    # 透传给 robust_v2 的其余参数（如 --render / --host / --port）放在 -- 之后
    p.add_argument("extra", nargs=argparse.REMAINDER,
                   help="-- 之后的参数原样透传给 train_no_rllib_mappo_robust_v2")
    args = p.parse_args()

    # ── 方式②：收编已有策略（优先；纯文件操作，不训练）──
    if args.from_ckpts:
        coop_ids = [int(x) for x in args.cooperative_ids.split(',')]
        print(f"[build_red_pool] 收编已有策略 → {args.out}（cooperative_ids={coop_ids}）")
        collected, failures = register_existing(
            args.from_ckpts, args.out, coop_ids, link=args.link, overwrite=args.overwrite)
        _print_summary(args.out, collected, failures)
        return

    # ── 方式①：重训多 seed ──
    extra = args.extra
    if extra and extra[0] == "--":
        extra = extra[1:]

    collected, failures = [], []
    for seed in args.seeds:
        try:
            if not args.collect_only:
                train_one_seed(seed, args.updates, args.base_logdir, extra)
            collected.append(collect_best(seed, args.base_logdir, args.out))
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"[build_red_pool] [!] seed {seed} 失败: {e}")
            failures.append((str(seed), str(e)))
    _print_summary(args.out, collected, failures)


if __name__ == "__main__":
    main()
