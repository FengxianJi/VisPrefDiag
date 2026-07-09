#!/usr/bin/env python3
import json
import numpy as np

# Load data
with open('NIMA.json', 'r') as f:
    data = json.load(f)

# Basic stats
n = len(data)
nima0 = [r['nima_mean_0'] for r in data]
nima1 = [r['nima_mean_1'] for r in data]

# Human agreement
correct = sum(1 for r in data if
    (r['nima_mean_0'] > r['nima_mean_1'] and r['label_0'] > r['label_1']) or
    (r['nima_mean_1'] > r['nima_mean_0'] and r['label_1'] > r['label_0']))

# Results
print(f"Total samples: {n}")
print(f"Image 0 mean: {np.mean(nima0):.4f} ± {np.std(nima0):.4f}")
print(f"Image 1 mean: {np.mean(nima1):.4f} ± {np.std(nima1):.4f}")
print(f"Overall mean: {np.mean(nima0 + nima1):.4f}")
print(f"Human agreement: {correct/n:.4f} ({correct/n*100:.1f}%)")

# Model stats
from collections import defaultdict
models = defaultdict(list)
for r in data:
    models[r['model_0']].append(r['nima_mean_0'])
    models[r['model_1']].append(r['nima_mean_1'])

print("\nBy model:")
for m in sorted(models.keys()):
    print(f"  {m}: {np.mean(models[m]):.4f}")
