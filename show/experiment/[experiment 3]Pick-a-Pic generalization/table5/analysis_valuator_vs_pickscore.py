#!/usr/bin/env python3
"""
Comprehensive analysis: Valuator (auction-based, zero-shot) vs PickScore (trained on Pick-a-Pic)
on the Pick-a-Pic test set.

Key context:
- PickScore is trained ON Pick-a-Pic data (home-field advantage)
- Valuator is a zero-shot auction-based mechanism using LLMs
- Valuator has ~61% tie rate (abstention), but when it decides, it's accurate
"""

import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats

# ============================================================
# Configuration
# ============================================================
BASE = "/mnt/data/user/fengxian.ji/AUCTION/show/table/table5"
META_BASE = "/mnt/data/user/fengxian.ji/AUCTION/show/experiment/[ICML][Mechanism-Based Value Elicitation ][2][26015]Human preference sanity /Generator_diff_instruction/sample"

SEEDS = [42, 2024, 7777]

PICKSCORE_PATH = os.path.join(BASE, "PickScore_v1.json")
CLIPSCORE_PATH = os.path.join(BASE, "CLIPScore.json")

def load_data():
    """Load and match all data sources."""
    # Load PickScore (full dataset, 13296 entries)
    with open(PICKSCORE_PATH) as f:
        ps_data = json.load(f)
    
    # Build PickScore lookup by ranking_id
    ps_by_ranking = {}
    for p in ps_data:
        rid = p['ranking_id']
        if rid not in ps_by_ranking:
            ps_by_ranking[rid] = p
    
    # Also build by (image_0_uid, image_1_uid) as fallback
    ps_by_uid = {}
    for p in ps_data:
        key = (p['image_0_uid'], p['image_1_uid'])
        if key not in ps_by_uid:
            ps_by_uid[key] = p
    
    all_seed_data = {}
    
    for seed in SEEDS:
        # Load Valuator seed data
        val_path = os.path.join(BASE, "Auction-arena", f"seed{seed}.json")
        with open(val_path) as f:
            val_data = json.load(f)
        
        # Load meta CSV for ranking_id mapping
        meta_path = os.path.join(META_BASE, f"pickapic_seed{seed}_n500_meta.csv")
        meta = pd.read_csv(meta_path)
        
        matched = []
        unmatched = 0
        
        for v in val_data:
            row_idx = v['row_idx']
            ps_entry = None
            
            # Primary: match via ranking_id from meta CSV
            if row_idx < len(meta):
                rid = meta.iloc[row_idx]['ranking_id']
                ps_entry = ps_by_ranking.get(rid)
            
            # Fallback: match via uid pair
            if ps_entry is None:
                key = (v['image_0_uid'], v['image_1_uid'])
                ps_entry = ps_by_uid.get(key)
            
            if ps_entry is not None:
                # Verify UIDs match
                if (v['image_0_uid'] == ps_entry['image_0_uid'] and 
                    v['image_1_uid'] == ps_entry['image_1_uid']):
                    matched.append({
                        'seed': seed,
                        'row_idx': row_idx,
                        'caption': v['caption'],
                        'image_0_uid': v['image_0_uid'],
                        'image_1_uid': v['image_1_uid'],
                        'model_0': v['model_0'],
                        'model_1': v['model_1'],
                        'label_0': v['label_0'],
                        'label_1': v['label_1'],
                        # Valuator fields
                        'val_price_0': v['price_0'],
                        'val_price_1': v['price_1'],
                        'val_is_tie': v['is_tie'],
                        'val_correct': v['correct'],
                        'val_pred': v.get('pred_preferred', -1),
                        'val_gt': v.get('gt_preferred', -1),
                        # PickScore fields
                        'ps_score_0': ps_entry['pickscore_0'],
                        'ps_score_1': ps_entry['pickscore_1'],
                    })
                else:
                    unmatched += 1
            else:
                unmatched += 1
        
        all_seed_data[seed] = matched
        print(f"Seed {seed}: {len(matched)} matched, {unmatched} unmatched")
    
    return all_seed_data


def compute_pickscore_prediction(entry):
    """Determine PickScore's prediction for a sample."""
    if entry['ps_score_0'] > entry['ps_score_1']:
        return 0  # prefers image 0
    elif entry['ps_score_1'] > entry['ps_score_0']:
        return 1  # prefers image 1
    else:
        return -1  # tie (extremely rare for continuous scores)


def compute_pickscore_correct(entry):
    """Check if PickScore's prediction is correct."""
    ps_pred = compute_pickscore_prediction(entry)
    if ps_pred == -1:
        return 0  # tie = wrong
    # Ground truth: the image with label=1.0 is preferred
    gt = 0 if entry['label_0'] == 1.0 else 1
    return 1 if ps_pred == gt else 0


def pickscore_margin(entry):
    """Absolute margin of PickScore scores."""
    return abs(entry['ps_score_0'] - entry['ps_score_1'])


# ============================================================
# Analysis Functions
# ============================================================

def analysis_basic_stats(all_data):
    """Print basic statistics per seed and overall."""
    print("\n" + "="*80)
    print("1. BASIC STATISTICS")
    print("="*80)
    
    total_val_correct = 0
    total_val_decided = 0
    total_ps_correct = 0
    total_n = 0
    total_ties = 0
    
    for seed in SEEDS:
        data = all_data[seed]
        n = len(data)
        ties = sum(1 for d in data if d['val_is_tie'] == 1)
        decided = n - ties
        val_correct = sum(1 for d in data if d['val_is_tie'] == 0 and d['val_correct'] == 1)
        ps_correct = sum(compute_pickscore_correct(d) for d in data)
        ps_correct_on_decided = sum(compute_pickscore_correct(d) for d in data if d['val_is_tie'] == 0)
        
        print(f"\n  Seed {seed} (n={n}, matched)")
        print(f"    Valuator: ties={ties} ({ties/n*100:.1f}%), decided={decided}")
        print(f"    Valuator accuracy (decided only): {val_correct}/{decided} = {val_correct/decided*100:.2f}%")
        print(f"    Valuator accuracy (all, ties=wrong): {val_correct}/{n} = {val_correct/n*100:.2f}%")
        print(f"    PickScore accuracy (all): {ps_correct}/{n} = {ps_correct/n*100:.2f}%")
        print(f"    PickScore accuracy (on Valuator's decided subset): {ps_correct_on_decided}/{decided} = {ps_correct_on_decided/decided*100:.2f}%")
        
        total_val_correct += val_correct
        total_val_decided += decided
        total_ps_correct += ps_correct
        total_n += n
        total_ties += ties
    
    print(f"\n  OVERALL (n={total_n})")
    print(f"    Valuator: ties={total_ties} ({total_ties/total_n*100:.1f}%), decided={total_val_decided}")
    print(f"    Valuator accuracy (decided only): {total_val_correct}/{total_val_decided} = {total_val_correct/total_val_decided*100:.2f}%")
    print(f"    Valuator accuracy (all, ties=wrong): {total_val_correct}/{total_n} = {total_val_correct/total_n*100:.2f}%")
    print(f"    PickScore accuracy (all): {total_ps_correct}/{total_n} = {total_ps_correct/total_n*100:.2f}%")


def analysis_head_to_head(all_data):
    """Head-to-head comparison on Valuator's DECIDED samples."""
    print("\n" + "="*80)
    print("2. HEAD-TO-HEAD COMPARISON (Valuator's decided samples only)")
    print("="*80)
    
    for seed in SEEDS:
        data = all_data[seed]
        decided = [d for d in data if d['val_is_tie'] == 0]
        
        both_correct = 0
        both_wrong = 0
        only_ps_correct = 0
        only_val_correct = 0
        
        for d in decided:
            vc = d['val_correct'] == 1
            pc = compute_pickscore_correct(d) == 1
            if vc and pc:
                both_correct += 1
            elif not vc and not pc:
                both_wrong += 1
            elif pc and not vc:
                only_ps_correct += 1
            elif vc and not pc:
                only_val_correct += 1
        
        n = len(decided)
        print(f"\n  Seed {seed} (n={n} decided samples)")
        print(f"    Both correct:          {both_correct:4d} ({both_correct/n*100:5.1f}%)")
        print(f"    Both wrong:            {both_wrong:4d} ({both_wrong/n*100:5.1f}%)")
        print(f"    Only PickScore correct:{only_ps_correct:4d} ({only_ps_correct/n*100:5.1f}%)")
        print(f"    Only Valuator correct: {only_val_correct:4d} ({only_val_correct/n*100:5.1f}%)")
        print(f"    Agreement rate: {(both_correct+both_wrong)/n*100:.1f}%")
        if (only_val_correct+both_wrong) > 0:
            print(f"    Valuator acc when PickScore WRONG: {only_val_correct}/{only_val_correct+both_wrong} = {only_val_correct/(only_val_correct+both_wrong)*100:.2f}%")
        if (only_ps_correct+both_wrong) > 0:
            print(f"    PickScore acc when Valuator WRONG: {only_ps_correct}/{only_ps_correct+both_wrong} = {only_ps_correct/(only_ps_correct+both_wrong)*100:.2f}%")
    
    # Aggregate
    print("\n  AGGREGATE across seeds:")
    all_decided = [d for seed in SEEDS for d in all_data[seed] if d['val_is_tie'] == 0]
    bc = sum(1 for d in all_decided if d['val_correct']==1 and compute_pickscore_correct(d)==1)
    bw = sum(1 for d in all_decided if d['val_correct']==0 and compute_pickscore_correct(d)==0)
    op = sum(1 for d in all_decided if d['val_correct']==0 and compute_pickscore_correct(d)==1)
    ov = sum(1 for d in all_decided if d['val_correct']==1 and compute_pickscore_correct(d)==0)
    n = len(all_decided)
    print(f"    n={n} decided samples total")
    print(f"    Both correct:          {bc:4d} ({bc/n*100:5.1f}%)")
    print(f"    Both wrong:            {bw:4d} ({bw/n*100:5.1f}%)")
    print(f"    Only PickScore correct:{op:4d} ({op/n*100:5.1f}%)")
    print(f"    Only Valuator correct: {ov:4d} ({ov/n*100:5.1f}%)")
    print(f"    Valuator acc when PickScore WRONG: {ov}/{ov+bw} = {ov/(ov+bw)*100:.2f}%")
    print(f"    PickScore acc when Valuator WRONG: {op}/{op+bw} = {op/(op+bw)*100:.2f}%")


def analysis_difficulty_stratification(all_data):
    """Stratify by PickScore margin (difficulty proxy)."""
    print("\n" + "="*80)
    print("3. DIFFICULTY STRATIFICATION (by PickScore margin)")
    print("="*80)
    
    # Pool all decided samples across seeds
    all_decided = []
    for seed in SEEDS:
        for d in all_data[seed]:
            if d['val_is_tie'] == 0:
                d_copy = dict(d)
                d_copy['ps_margin'] = pickscore_margin(d)
                d_copy['ps_correct'] = compute_pickscore_correct(d)
                all_decided.append(d_copy)
    
    margins = [d['ps_margin'] for d in all_decided]
    
    # Method 1: Tercile-based bins
    terciles = np.percentile(margins, [33.33, 66.67])
    print(f"\n  PickScore margin distribution (decided samples, n={len(all_decided)}):")
    print(f"    Min: {min(margins):.4f}, Max: {max(margins):.4f}, Mean: {np.mean(margins):.4f}, Median: {np.median(margins):.4f}")
    print(f"    Tercile boundaries: {terciles[0]:.4f}, {terciles[1]:.4f}")
    
    bins = [
        ("Hard (low margin)", lambda m: m <= terciles[0]),
        ("Medium", lambda m: terciles[0] < m <= terciles[1]),
        ("Easy (high margin)", lambda m: m > terciles[1]),
    ]
    
    print(f"\n  {'Difficulty':<25s} {'N':>5s} {'Val Acc':>10s} {'PS Acc':>10s} {'Diff':>10s}")
    print("  " + "-"*65)
    
    for label, cond in bins:
        subset = [d for d in all_decided if cond(d['ps_margin'])]
        n = len(subset)
        if n == 0:
            continue
        val_acc = sum(d['val_correct'] for d in subset) / n * 100
        ps_acc = sum(d['ps_correct'] for d in subset) / n * 100
        diff = val_acc - ps_acc
        print(f"  {label:<25s} {n:>5d} {val_acc:>9.2f}% {ps_acc:>9.2f}% {diff:>+9.2f}%")
    
    # Also show ALL samples (not just decided) for context
    print(f"\n  Including ALL samples (ties count as wrong for Valuator):")
    all_samples = []
    for seed in SEEDS:
        for d in all_data[seed]:
            d_copy = dict(d)
            d_copy['ps_margin'] = pickscore_margin(d)
            d_copy['ps_correct'] = compute_pickscore_correct(d)
            all_samples.append(d_copy)
    
    margins_all = [d['ps_margin'] for d in all_samples]
    terciles_all = np.percentile(margins_all, [33.33, 66.67])
    
    bins_all = [
        ("Hard (low margin)", lambda m: m <= terciles_all[0]),
        ("Medium", lambda m: terciles_all[0] < m <= terciles_all[1]),
        ("Easy (high margin)", lambda m: m > terciles_all[1]),
    ]
    
    print(f"\n  {'Difficulty':<25s} {'N':>5s} {'Val Acc(dec)':>12s} {'Val Tie%':>10s} {'PS Acc':>10s}")
    print("  " + "-"*65)
    
    for label, cond in bins_all:
        subset = [d for d in all_samples if cond(d['ps_margin'])]
        n = len(subset)
        if n == 0:
            continue
        decided_sub = [d for d in subset if d['val_is_tie'] == 0]
        n_decided = len(decided_sub)
        val_correct = sum(d['val_correct'] for d in decided_sub) if n_decided > 0 else 0
        tie_rate = (n - n_decided) / n * 100
        val_acc = val_correct / n_decided * 100 if n_decided > 0 else 0
        ps_acc = sum(d['ps_correct'] for d in subset) / n * 100
        print(f"  {label:<25s} {n:>5d} {val_acc:>11.2f}% {tie_rate:>9.1f}% {ps_acc:>9.2f}%")


def analysis_by_model_pairs(all_data):
    """Group by generation model pairs."""
    print("\n" + "="*80)
    print("4. BY GENERATION MODEL PAIRS")
    print("="*80)
    
    # Pool all decided samples
    pair_stats = defaultdict(lambda: {'val_correct': 0, 'ps_correct': 0, 'n': 0, 'total': 0, 'ties': 0})
    
    for seed in SEEDS:
        for d in all_data[seed]:
            pair = tuple(sorted([d['model_0'], d['model_1']]))
            pair_stats[pair]['total'] += 1
            if d['val_is_tie'] == 1:
                pair_stats[pair]['ties'] += 1
                continue
            pair_stats[pair]['n'] += 1
            pair_stats[pair]['val_correct'] += d['val_correct']
            pair_stats[pair]['ps_correct'] += compute_pickscore_correct(d)
    
    print(f"\n  {'Model Pair':<35s} {'Total':>6s} {'Dec':>5s} {'Tie%':>7s} {'Val Acc':>10s} {'PS Acc':>10s} {'Diff':>10s}")
    print("  " + "-"*90)
    
    # Sort by number of decided samples
    for pair, s in sorted(pair_stats.items(), key=lambda x: -x[1]['n']):
        n = s['n']
        total = s['total']
        ties = s['ties']
        if n < 5:
            continue
        val_acc = s['val_correct'] / n * 100
        ps_acc = s['ps_correct'] / n * 100
        diff = val_acc - ps_acc
        pair_str = f"{pair[0]} vs {pair[1]}"
        tie_rate = ties / total * 100
        print(f"  {pair_str:<35s} {total:>6d} {n:>5d} {tie_rate:>6.1f}% {val_acc:>9.2f}% {ps_acc:>9.2f}% {diff:>+9.2f}%")
    
    # Find pairs where Valuator beats PickScore
    print("\n  Model pairs where Valuator BEATS PickScore (decided, n>=5):")
    beats = [(pair, s) for pair, s in pair_stats.items() 
             if s['n'] >= 5 and s['val_correct']/s['n'] > s['ps_correct']/s['n']]
    if beats:
        for pair, s in sorted(beats, key=lambda x: -(x[1]['val_correct']/x[1]['n'] - x[1]['ps_correct']/x[1]['n'])):
            pair_str = f"{pair[0]} vs {pair[1]}"
            val_acc = s['val_correct']/s['n']*100
            ps_acc = s['ps_correct']/s['n']*100
            print(f"    {pair_str}: Val {val_acc:.1f}% vs PS {ps_acc:.1f}% (n={s['n']} decided)")
    else:
        print("    None found")


def analysis_complementarity(all_data):
    """Oracle accuracy if using best of PickScore and Valuator per sample."""
    print("\n" + "="*80)
    print("5. COMPLEMENTARITY ANALYSIS")
    print("="*80)
    
    for seed in SEEDS:
        data = all_data[seed]
        n = len(data)
        decided = [d for d in data if d['val_is_tie'] == 0]
        
        # On decided samples
        n_dec = len(decided)
        ps_only = sum(compute_pickscore_correct(d) for d in decided)
        val_only = sum(d['val_correct'] for d in decided)
        oracle = sum(1 for d in decided if d['val_correct']==1 or compute_pickscore_correct(d)==1)
        
        print(f"\n  Seed {seed} - Decided samples (n={n_dec}):")
        print(f"    PickScore alone:   {ps_only}/{n_dec} = {ps_only/n_dec*100:.2f}%")
        print(f"    Valuator alone:    {val_only}/{n_dec} = {val_only/n_dec*100:.2f}%")
        print(f"    Oracle (best of):  {oracle}/{n_dec} = {oracle/n_dec*100:.2f}%")
        print(f"    Improvement over PickScore: +{(oracle-ps_only)/n_dec*100:.2f}pp")
        print(f"    Improvement over Valuator:  +{(oracle-val_only)/n_dec*100:.2f}pp")
    
    # Aggregate
    all_decided = [d for seed in SEEDS for d in all_data[seed] if d['val_is_tie'] == 0]
    n = len(all_decided)
    ps_c = sum(compute_pickscore_correct(d) for d in all_decided)
    val_c = sum(d['val_correct'] for d in all_decided)
    orc = sum(1 for d in all_decided if d['val_correct']==1 or compute_pickscore_correct(d)==1)
    print(f"\n  AGGREGATE - Decided samples (n={n}):")
    print(f"    PickScore alone:   {ps_c}/{n} = {ps_c/n*100:.2f}%")
    print(f"    Valuator alone:    {val_c}/{n} = {val_c/n*100:.2f}%")
    print(f"    Oracle (best of):  {orc}/{n} = {orc/n*100:.2f}%")
    print(f"    Improvement over PickScore: +{(orc-ps_c)/n*100:.2f}pp")
    print(f"    Improvement over Valuator:  +{(orc-val_c)/n*100:.2f}pp")
    
    # Strategy analysis: Use Valuator when it decides, PickScore when Valuator abstains
    print(f"\n  STRATEGY: 'Valuator when decided, PickScore when tie':")
    for seed in SEEDS:
        data = all_data[seed]
        n = len(data)
        correct = 0
        for d in data:
            if d['val_is_tie'] == 0:
                correct += d['val_correct']
            else:
                correct += compute_pickscore_correct(d)
        ps_all = sum(compute_pickscore_correct(d) for d in data)
        print(f"    Seed {seed}: {correct}/{n} = {correct/n*100:.2f}%  (vs PickScore alone: {ps_all/n*100:.2f}%)")
    
    all_data_flat = [d for seed in SEEDS for d in all_data[seed]]
    n_all = len(all_data_flat)
    correct_combo = sum(d['val_correct'] if d['val_is_tie']==0 else compute_pickscore_correct(d) for d in all_data_flat)
    ps_all_total = sum(compute_pickscore_correct(d) for d in all_data_flat)
    print(f"    AGGREGATE: {correct_combo}/{n_all} = {correct_combo/n_all*100:.2f}%  (vs PickScore alone: {ps_all_total/n_all*100:.2f}%)")


def analysis_statistical_significance(all_data):
    """McNemar's test comparing Valuator vs PickScore on decided samples."""
    print("\n" + "="*80)
    print("6. STATISTICAL SIGNIFICANCE")
    print("="*80)
    
    for seed in SEEDS:
        decided = [d for d in all_data[seed] if d['val_is_tie'] == 0]
        
        # Contingency table for McNemar's test
        # b = PickScore correct, Valuator wrong
        # c = Valuator correct, PickScore wrong
        b = sum(1 for d in decided if compute_pickscore_correct(d)==1 and d['val_correct']==0)
        c = sum(1 for d in decided if d['val_correct']==1 and compute_pickscore_correct(d)==0)
        
        # McNemar's test (with continuity correction)
        if b + c > 0:
            chi2 = (abs(b - c) - 1)**2 / (b + c)
            p_value = 1 - stats.chi2.cdf(chi2, df=1)
            # Exact binomial test
            p_exact = stats.binomtest(b, b + c, 0.5).pvalue
        else:
            chi2 = 0
            p_value = 1.0
            p_exact = 1.0
        
        print(f"\n  Seed {seed} (n={len(decided)} decided):")
        print(f"    Discordant pairs: PickScore-only correct (b)={b}, Valuator-only correct (c)={c}")
        print(f"    McNemar chi2={chi2:.4f}, p={p_value:.4f}")
        print(f"    Exact binomial p={p_exact:.4f}")
        if p_value < 0.05:
            winner = "PickScore" if b > c else "Valuator"
            print(f"    => Significant difference (p<0.05), {winner} better on decided samples")
        else:
            print(f"    => No significant difference (p>=0.05) on decided samples")
    
    # Aggregate
    all_decided = [d for seed in SEEDS for d in all_data[seed] if d['val_is_tie'] == 0]
    b_all = sum(1 for d in all_decided if compute_pickscore_correct(d)==1 and d['val_correct']==0)
    c_all = sum(1 for d in all_decided if d['val_correct']==1 and compute_pickscore_correct(d)==0)
    
    if b_all + c_all > 0:
        chi2_all = (abs(b_all - c_all) - 1)**2 / (b_all + c_all)
        p_all = 1 - stats.chi2.cdf(chi2_all, df=1)
        p_exact_all = stats.binomtest(b_all, b_all + c_all, 0.5).pvalue
    else:
        chi2_all = 0
        p_all = 1.0
        p_exact_all = 1.0
    
    print(f"\n  AGGREGATE (n={len(all_decided)} decided):")
    print(f"    Discordant: b={b_all} (PS-only correct), c={c_all} (Val-only correct)")
    print(f"    McNemar chi2={chi2_all:.4f}, p={p_all:.4f}")
    print(f"    Exact binomial p={p_exact_all:.4f}")


def analysis_selective_prediction(all_data):
    """Selective prediction: compare accuracy at various coverage levels."""
    print("\n" + "="*80)
    print("7. SELECTIVE PREDICTION ANALYSIS")
    print("="*80)
    
    # Pool all samples across seeds
    all_samples = []
    for seed in SEEDS:
        for d in all_data[seed]:
            entry = dict(d)
            entry['ps_correct'] = compute_pickscore_correct(d)
            entry['ps_margin'] = pickscore_margin(d)
            entry['val_margin'] = abs(d['val_price_0'] - d['val_price_1'])
            all_samples.append(entry)
    
    n_total = len(all_samples)
    
    # ---- Valuator's selective prediction ----
    # Valuator naturally abstains (tie). Sort non-tie by margin for further selectivity
    decided = [d for d in all_samples if d['val_is_tie'] == 0]
    ties = [d for d in all_samples if d['val_is_tie'] == 1]
    
    # Sort decided by val_margin (descending = most confident first)
    decided_sorted = sorted(decided, key=lambda d: -d['val_margin'])
    
    # ---- PickScore's selective prediction ----
    # Sort all by ps_margin (descending = most confident first)
    ps_sorted = sorted(all_samples, key=lambda d: -d['ps_margin'])
    
    # Coverage levels to evaluate
    coverage_levels = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    print(f"\n  Total samples: {n_total}")
    print(f"  Valuator decided: {len(decided)} ({len(decided)/n_total*100:.1f}%)")
    print(f"  Valuator ties: {len(ties)} ({len(ties)/n_total*100:.1f}%)")
    
    print(f"\n  {'Coverage':>10s} {'Val Acc':>10s} {'Val N':>7s} {'PS Acc':>10s} {'PS N':>7s} {'Val-PS':>10s}")
    print("  " + "-"*60)
    
    crossover_found = False
    crossover_coverage = None
    
    for cov in coverage_levels:
        n_select = int(n_total * cov)
        
        # Valuator: select top n_select by margin (from decided samples)
        val_selected = decided_sorted[:min(n_select, len(decided_sorted))]
        val_n = len(val_selected)
        val_acc = sum(d['val_correct'] for d in val_selected) / val_n * 100 if val_n > 0 else 0
        
        # PickScore: select top n_select by margin
        ps_selected = ps_sorted[:n_select]
        ps_n = len(ps_selected)
        ps_acc = sum(d['ps_correct'] for d in ps_selected) / ps_n * 100 if ps_n > 0 else 0
        
        diff = val_acc - ps_acc
        marker = " <-- Valuator better" if diff > 0 else ""
        print(f"  {cov:>9.0%} {val_acc:>9.2f}% {val_n:>7d} {ps_acc:>9.2f}% {ps_n:>7d} {diff:>+9.2f}%{marker}")
        
        if diff > 0 and not crossover_found:
            crossover_found = True
            crossover_coverage = cov
    
    if crossover_found:
        print(f"\n  Valuator exceeds PickScore at coverage <= {crossover_coverage:.0%}")
    
    # More granular near the natural abstention point
    print(f"\n  Fine-grained around Valuator's natural coverage ({len(decided)/n_total*100:.1f}%):")
    
    val_natural_cov = len(decided) / n_total
    fine_coverages = np.arange(0.05, min(val_natural_cov + 0.15, 1.01), 0.02)
    
    print(f"\n  {'Coverage':>10s} {'Val Acc':>10s} {'Val N':>7s} {'PS Acc':>10s} {'PS N':>7s} {'Val-PS':>10s}")
    print("  " + "-"*60)
    
    for cov in fine_coverages:
        n_select = int(n_total * cov)
        
        val_selected = decided_sorted[:min(n_select, len(decided_sorted))]
        val_n = len(val_selected)
        val_acc = sum(d['val_correct'] for d in val_selected) / val_n * 100 if val_n > 0 else 0
        
        ps_selected = ps_sorted[:n_select]
        ps_n = len(ps_selected)
        ps_acc = sum(d['ps_correct'] for d in ps_selected) / ps_n * 100 if ps_n > 0 else 0
        
        diff = val_acc - ps_acc
        marker = " <--" if diff > 0 else ""
        print(f"  {cov:>9.0%} {val_acc:>9.2f}% {val_n:>7d} {ps_acc:>9.2f}% {ps_n:>7d} {diff:>+9.2f}%{marker}")
    
    # ---- Accuracy-Rejection Curve (ARC) summary ----
    print(f"\n  Accuracy-Rejection Curve (ARC) - Area Under Curve:")
    
    # Compute AUC for both methods
    val_aurc = 0
    ps_aurc = 0
    prev_cov = 0
    for cov in np.arange(0.01, 1.01, 0.01):
        n_select = max(1, int(n_total * cov))
        
        val_sel = decided_sorted[:min(n_select, len(decided_sorted))]
        val_n = len(val_sel)
        val_acc = sum(d['val_correct'] for d in val_sel) / val_n if val_n > 0 else 0
        
        ps_sel = ps_sorted[:n_select]
        ps_acc = sum(d['ps_correct'] for d in ps_sel) / len(ps_sel) if ps_sel else 0
        
        step = cov - prev_cov
        # Only integrate where Valuator has coverage
        if val_n > 0:
            val_aurc += val_acc * step
        ps_aurc += ps_acc * step
        prev_cov = cov
    
    print(f"    Valuator AUARC (capped at natural coverage): {val_aurc:.4f}")
    print(f"    PickScore AUARC (full): {ps_aurc:.4f}")


def analysis_per_seed_summary(all_data):
    """Compact per-seed summary table."""
    print("\n" + "="*80)
    print("8. COMPACT SUMMARY TABLE")
    print("="*80)
    
    header = f"{'Metric':<50s}"
    for seed in SEEDS:
        header += f" {'Seed '+str(seed):>12s}"
    header += f" {'Average':>12s}"
    print(f"\n  {header}")
    print("  " + "-"*100)
    
    for metric_name, compute_fn in [
        ("Total samples", lambda data: len(data)),
        ("Valuator tie rate (%)", lambda data: sum(d['val_is_tie'] for d in data) / len(data) * 100),
        ("Valuator acc - decided (%)", lambda data: sum(d['val_correct'] for d in data if d['val_is_tie']==0) / max(1, sum(1 for d in data if d['val_is_tie']==0)) * 100),
        ("PickScore acc - all (%)", lambda data: sum(compute_pickscore_correct(d) for d in data) / len(data) * 100),
        ("PickScore acc - on decided subset (%)", lambda data: sum(compute_pickscore_correct(d) for d in data if d['val_is_tie']==0) / max(1, sum(1 for d in data if d['val_is_tie']==0)) * 100),
        ("Valuator acc when PS wrong - dec (%)", lambda data: (
            sum(1 for d in data if d['val_is_tie']==0 and d['val_correct']==1 and compute_pickscore_correct(d)==0) / 
            max(1, sum(1 for d in data if d['val_is_tie']==0 and compute_pickscore_correct(d)==0)) * 100
        )),
        ("Oracle best-of - decided (%)", lambda data: sum(1 for d in data if d['val_is_tie']==0 and (d['val_correct']==1 or compute_pickscore_correct(d)==1)) / max(1, sum(1 for d in data if d['val_is_tie']==0)) * 100),
        ("Combo strategy acc (%)", lambda data: sum(d['val_correct'] if d['val_is_tie']==0 else compute_pickscore_correct(d) for d in data) / len(data) * 100),
    ]:
        row = f"  {metric_name:<50s}"
        vals = []
        for seed in SEEDS:
            v = compute_fn(all_data[seed])
            vals.append(v)
            if isinstance(v, float):
                row += f" {v:>11.2f}%"
            else:
                row += f" {v:>12d}"
        avg = np.mean(vals)
        if isinstance(vals[0], float):
            row += f" {avg:>11.2f}%"
        else:
            row += f" {avg:>12.1f}"
        print(row)


def analysis_price_margin_correlation(all_data):
    """Analyze relationship between Valuator's price margin and correctness."""
    print("\n" + "="*80)
    print("9. VALUATOR CONFIDENCE CALIBRATION")
    print("="*80)
    
    all_decided = []
    for seed in SEEDS:
        for d in all_data[seed]:
            if d['val_is_tie'] == 0:
                d_copy = dict(d)
                d_copy['val_margin'] = abs(d['val_price_0'] - d['val_price_1'])
                all_decided.append(d_copy)
    
    margins = [d['val_margin'] for d in all_decided]
    corrects = [d['val_correct'] for d in all_decided]
    
    # Point-biserial correlation
    corr, p_val = stats.pointbiserialr(corrects, margins)
    print(f"\n  Point-biserial correlation (correct vs margin): r={corr:.4f}, p={p_val:.4f}")
    
    # Bin by margin quartiles
    quartiles = np.percentile(margins, [25, 50, 75])
    print(f"  Margin quartiles: {quartiles}")
    
    bins = [
        ("Q1 (smallest margin)", lambda m: m <= quartiles[0]),
        ("Q2", lambda m: quartiles[0] < m <= quartiles[1]),
        ("Q3", lambda m: quartiles[1] < m <= quartiles[2]),
        ("Q4 (largest margin)", lambda m: m > quartiles[2]),
    ]
    
    print(f"\n  {'Quartile':<25s} {'N':>5s} {'Acc':>10s} {'Avg Margin':>12s}")
    print("  " + "-"*55)
    
    for label, cond in bins:
        subset = [d for d in all_decided if cond(d['val_margin'])]
        n = len(subset)
        if n == 0:
            continue
        acc = sum(d['val_correct'] for d in subset) / n * 100
        avg_m = np.mean([d['val_margin'] for d in subset])
        print(f"  {label:<25s} {n:>5d} {acc:>9.2f}% {avg_m:>11.2f}")


def analysis_tie_characteristics(all_data):
    """What kinds of samples does Valuator tie on? Are they truly harder?"""
    print("\n" + "="*80)
    print("10. TIE ANALYSIS: Are Valuator's ties truly harder?")
    print("="*80)
    
    decided_ps_margins = []
    tie_ps_margins = []
    decided_ps_accs = []
    tie_ps_accs = []
    
    for seed in SEEDS:
        for d in all_data[seed]:
            margin = pickscore_margin(d)
            ps_correct = compute_pickscore_correct(d)
            if d['val_is_tie'] == 0:
                decided_ps_margins.append(margin)
                decided_ps_accs.append(ps_correct)
            else:
                tie_ps_margins.append(margin)
                tie_ps_accs.append(ps_correct)
    
    print(f"\n  PickScore margin on Valuator's DECIDED samples:")
    print(f"    N={len(decided_ps_margins)}, Mean={np.mean(decided_ps_margins):.4f}, Median={np.median(decided_ps_margins):.4f}")
    print(f"    PickScore accuracy: {sum(decided_ps_accs)/len(decided_ps_accs)*100:.2f}%")
    
    print(f"\n  PickScore margin on Valuator's TIE samples:")
    print(f"    N={len(tie_ps_margins)}, Mean={np.mean(tie_ps_margins):.4f}, Median={np.median(tie_ps_margins):.4f}")
    print(f"    PickScore accuracy: {sum(tie_ps_accs)/len(tie_ps_accs)*100:.2f}%")
    
    # t-test on margins
    t_stat, p_val = stats.ttest_ind(decided_ps_margins, tie_ps_margins)
    print(f"\n  t-test (decided vs tie margins): t={t_stat:.4f}, p={p_val:.4f}")
    if p_val < 0.05:
        higher = "decided" if np.mean(decided_ps_margins) > np.mean(tie_ps_margins) else "tie"
        print(f"    => Significant: {higher} samples have higher PickScore margin")
    else:
        print(f"    => No significant difference in PickScore margin")
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pval = stats.mannwhitneyu(decided_ps_margins, tie_ps_margins, alternative='two-sided')
    print(f"  Mann-Whitney U test: U={u_stat:.1f}, p={u_pval:.4f}")
    
    # Key insight
    ps_acc_decided = sum(decided_ps_accs)/len(decided_ps_accs)*100
    ps_acc_ties = sum(tie_ps_accs)/len(tie_ps_accs)*100
    print(f"\n  KEY INSIGHT:")
    print(f"    PickScore accuracy on Valuator's decided samples: {ps_acc_decided:.2f}%")
    print(f"    PickScore accuracy on Valuator's tie samples:     {ps_acc_ties:.2f}%")
    print(f"    Difference: {ps_acc_decided - ps_acc_ties:+.2f}pp")
    if ps_acc_decided > ps_acc_ties:
        print(f"    => PickScore is ALSO worse on samples Valuator abstains on.")
        print(f"       This validates Valuator's abstention as identifying genuinely hard cases.")
    else:
        print(f"    => PickScore performs similarly or better on tie samples.")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("="*80)
    print("COMPREHENSIVE ANALYSIS: Valuator (zero-shot, auction) vs PickScore (trained)")
    print("Dataset: Pick-a-Pic test set (PickScore's home turf)")
    print("="*80)
    
    all_data = load_data()
    
    analysis_basic_stats(all_data)
    analysis_head_to_head(all_data)
    analysis_difficulty_stratification(all_data)
    analysis_by_model_pairs(all_data)
    analysis_complementarity(all_data)
    analysis_statistical_significance(all_data)
    analysis_selective_prediction(all_data)
    analysis_per_seed_summary(all_data)
    analysis_price_margin_correlation(all_data)
    analysis_tie_characteristics(all_data)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
