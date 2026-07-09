#!/usr/bin/env python3
"""
Stratified diversity sampling of Pick-a-Pic (test) dataset.

Creates 3 subsets of 500 unique data pairs (rows) each.
No row overlap between files. Diversity ensured via stratification.

Usage:
    python sample_pickapic.py
"""

import os
import glob
import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = "/mnt/data/user/fengxian.ji/AUCTION/benchmark/Pick-a-Pic(test)"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

SEEDS = [42, 2024, 7777]
N_SAMPLES = 500

META_COLS = [
    "caption", "label_0", "label_1", "model_0", "model_1",
    "has_label", "are_different", "best_image_uid",
    "image_0_uid", "image_1_uid", "ranking_id",
]


# ── Helper functions ───────────────────────────────────────────────────────
def assign_caption_length_bin(caption_lengths: pd.Series) -> pd.Series:
    return pd.cut(
        caption_lengths, bins=[0, 30, 80, float("inf")],
        labels=["short", "medium", "long"], include_lowest=True,
    )

def assign_preference_bin(label_0: pd.Series) -> pd.Series:
    return pd.cut(
        label_0, bins=[-0.01, 0.3, 0.7, 1.01],
        labels=["strong_1", "weak", "strong_0"], include_lowest=True,
    )

def stratified_sample(df: pd.DataFrame, n: int, rng: np.random.RandomState) -> pd.DataFrame:
    df = df.copy()
    df["_caption_len"] = df["caption"].str.len().fillna(0)
    df["_len_bin"] = assign_caption_length_bin(df["_caption_len"])
    df["_pref_bin"] = assign_preference_bin(df["label_0"])
    df["_model_pair"] = df["model_0"] + " vs " + df["model_1"]
    df["_stratum"] = (
        df["_model_pair"].astype(str) + "|"
        + df["_len_bin"].astype(str) + "|"
        + df["_pref_bin"].astype(str)
    )

    stratum_counts = df["_stratum"].value_counts()
    total = len(df)
    alloc = (stratum_counts / total * n).apply(np.floor).astype(int).clip(lower=1)
    while alloc.sum() > n:
        alloc[alloc.idxmax()] -= 1

    sampled_indices = []
    for stratum, count in alloc.items():
        pool = df[df["_stratum"] == stratum].index.tolist()
        k = min(count, len(pool))
        sampled_indices.extend(rng.choice(pool, size=k, replace=False).tolist())

    remaining = n - len(sampled_indices)
    if remaining > 0:
        leftover = df.index.difference(pd.Index(sampled_indices)).tolist()
        sampled_indices.extend(rng.choice(leftover, size=min(remaining, len(leftover)), replace=False).tolist())

    return df.loc[sampled_indices].drop(
        columns=["_caption_len", "_len_bin", "_pref_bin", "_model_pair", "_stratum"]
    )

def print_report(df: pd.DataFrame, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}  —  {len(df)} samples")
    print(f"{'='*60}")
    mp = (df["model_0"] + " vs " + df["model_1"]).value_counts()
    print(f"\n  Model pairs ({len(mp)} unique):")
    for pair, cnt in mp.items():
        print(f"    {pair:40s} {cnt:4d}  ({cnt/len(df)*100:5.1f}%)")
    cap_len = df["caption"].str.len().fillna(0)
    bins = assign_caption_length_bin(cap_len).value_counts().sort_index()
    print(f"\n  Caption length:")
    for b, cnt in bins.items():
        print(f"    {str(b):10s} {cnt:4d}  ({cnt/len(df)*100:5.1f}%)")
    pref = assign_preference_bin(df["label_0"]).value_counts().sort_index()
    print(f"\n  Preference:")
    for p, cnt in pref.items():
        print(f"    {str(p):10s} {cnt:4d}  ({cnt/len(df)*100:5.1f}%)")
    print(f"\n  Unique captions: {df['caption'].nunique()}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parquet_files = sorted(glob.glob(os.path.join(DATA_DIR, "test-*.parquet")))

    # Step 1: Load metadata
    print("Loading metadata...")
    meta_parts = []
    shard_info = []
    offset = 0
    for f in parquet_files:
        part = pd.read_parquet(f, columns=META_COLS)
        n = len(part)
        meta_parts.append(part)
        shard_info.append((f, offset, n))
        offset += n
        print(f"  {os.path.basename(f)}: {n} rows")

    meta = pd.concat(meta_parts, ignore_index=True)
    meta["_shard_file"] = ""
    meta["_local_idx"] = -1
    for f, start, n in shard_info:
        meta.loc[start : start + n - 1, "_shard_file"] = f
        meta.loc[start : start + n - 1, "_local_idx"] = range(n)

    # Step 2: Filter to clean data
    clean = meta[(meta["has_label"] == True) & (meta["are_different"] == True)].copy()
    print(f"\nTotal: {len(meta)} | Clean: {len(clean)} | Need: 3 x {N_SAMPLES} = {3*N_SAMPLES}")

    # Step 3: Sample 3 x 500 with no overlap
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    used_indices = set()

    for i, seed in enumerate(SEEDS):
        print(f"\n{'#'*60}")
        print(f"  File {i+1}/3  —  seed={seed}")
        print(f"{'#'*60}")

        pool = clean.drop(index=list(used_indices), errors="ignore")
        print(f"  Pool: {len(pool)} rows")

        rng = np.random.RandomState(seed)
        sampled = stratified_sample(pool, N_SAMPLES, rng)
        used_indices.update(sampled.index.tolist())

        print_report(sampled, f"Seed {seed}")

        # Extract full rows with images
        grouped = sampled.groupby("_shard_file")["_local_idx"].apply(list).to_dict()
        print(f"\n  Extracting full rows...")
        full_parts = []
        for shard_file, local_indices in grouped.items():
            shard_df = pd.read_parquet(shard_file)
            full_parts.append(shard_df.iloc[local_indices])
            print(f"    {os.path.basename(shard_file)}: {len(local_indices)} rows")
            del shard_df

        full_df = pd.concat(full_parts, ignore_index=True)

        out_path = os.path.join(OUTPUT_DIR, f"pickapic_seed{seed}_n{N_SAMPLES}.parquet")
        full_df.to_parquet(out_path, index=False)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  Saved: {out_path}  ({size_mb:.1f} MB)")

        csv_path = os.path.join(OUTPUT_DIR, f"pickapic_seed{seed}_n{N_SAMPLES}_meta.csv")
        full_df.drop(columns=["jpg_0", "jpg_1"], errors="ignore").to_csv(csv_path, index=False)
        print(f"  CSV:   {csv_path}")
        del full_df, full_parts

    # Verification
    print(f"\n{'='*60}")
    print(f"  Total unique rows used: {len(used_indices)}  (expected {3*N_SAMPLES})")
    assert len(used_indices) == 3 * N_SAMPLES, "OVERLAP DETECTED!"
    print("  No overlap confirmed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
