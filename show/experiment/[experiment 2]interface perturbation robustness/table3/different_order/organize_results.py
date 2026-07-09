#!/usr/bin/env python3
"""
整理 Disordertheorder 目录结构：
将每个 disorder_* 根目录下的 metrics.json 和 test_original.json
移入 Single-agentLLM_score/ 子目录，使结构与其他模型 (Auction-arena, Valuator_judge 等) 一致。

整理前:
  disorder_1/
    metrics.json              ← run_score.py 产生 (Single-agentLLM_score)
    test_original.json        ← run_score.py 产生
    Auction-arena/metrics.json + test_original.json
    CLIPScore/metrics.json + CLIPScore.json
    ...

整理后:
  disorder_1/
    Single-agentLLM_score/metrics.json + test_original.json   ← 新建子目录
    Auction-arena/metrics.json + test_original.json
    CLIPScore/metrics.json + CLIPScore.json
    ...
"""

import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
TARGET_SUBDIR = "Single-agentLLM_score"
NUM_DISORDERS = 4


def organize():
    for d in range(1, NUM_DISORDERS + 1):
        disorder_dir = BASE_DIR / f"disorder_{d}"
        if not disorder_dir.exists():
            print(f"[SKIP] {disorder_dir} does not exist")
            continue

        target_dir = disorder_dir / TARGET_SUBDIR
        target_dir.mkdir(exist_ok=True)

        moved = []
        for fname in ["metrics.json", "test_original.json"]:
            src = disorder_dir / fname
            dst = target_dir / fname
            if src.exists():
                if dst.exists():
                    print(f"  [WARN] {dst} already exists, overwriting")
                shutil.move(str(src), str(dst))
                moved.append(fname)

        if moved:
            print(f"[disorder_{d}] Moved {moved} -> {TARGET_SUBDIR}/")
        else:
            print(f"[disorder_{d}] No root-level files to move")

    print("\nDone! Final structure:")
    for d in range(1, NUM_DISORDERS + 1):
        disorder_dir = BASE_DIR / f"disorder_{d}"
        if not disorder_dir.exists():
            continue
        subdirs = sorted([p.name for p in disorder_dir.iterdir() if p.is_dir()])
        root_files = sorted([p.name for p in disorder_dir.iterdir() if p.is_file()])
        print(f"\n  disorder_{d}/")
        if root_files:
            for f in root_files:
                print(f"    {f}")
        for sd in subdirs:
            files = sorted([p.name for p in (disorder_dir / sd).iterdir()])
            print(f"    {sd}/  ({', '.join(files)})")


if __name__ == "__main__":
    organize()
