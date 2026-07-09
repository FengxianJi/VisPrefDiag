# Benchmark

This folder keeps benchmark metadata and lightweight helpers used by the VisPrefDiag experiments.

## ImageReward

Included:

- `test.json`
- `test_1.json`
- `test_2.json`
- `test_with_alternatives.json`
- `test_1_with_alternatives.json`
- `test_2_with_alternatives.json`
- `refl_data.json`
- `metadata-test.parquet`
- `metadata-validation.parquet`

Excluded:

- raw `.webp` images
- `test_1.zip`
- `test_2.zip`
- the full `ImageReward.zip`

Those large assets remain in `../Valuator/benchmark/ImageReward/`.

## Pick-a-Pic

`Pick-a-Pic(test)/example_usage.py` and `download_pickapic.sh` are retained as lightweight dataset helpers. The sampled rows used for Table 4 live with the experiment under `show/experiment/[experiment 3]Pick-a-Pic generalization/`.
