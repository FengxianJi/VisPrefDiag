# Qwen3.5-27B migration package

This package contains the portable evaluation logic and the current append-only
JSONL checkpoints for the three PrefHardBench protocols:

- `single_score`: scores each image independently and derives A/B preference;
- `pairwise`: evaluates the original and swapped image order;
- `evidence`: evaluates both orders with evidence-first prompts.

The included checkpoints are partial and are intended for resume, not as final
results. The runner skips every `question_id` whose existing record has
`success: true`; failed records are retried on the next run.

## Requirements

The destination server needs:

1. This repository (or this package) and the `PrefHardBench` directory;
2. `PrefHardBench/data_annotations.csv` and `PrefHardBench/images/`;
3. Python 3.10+ and the package in `requirements.txt`;
4. A key file supplied outside Git, with one Qwen-compatible API key per line.

The key file is deliberately not included in this package. Do not commit it.

```bash
python3 -m pip install -r experiments/qwen35_27b_migration/requirements.txt
chmod +x experiments/qwen35_27b_migration/run_all.sh
```

## Run on another server

From the repository root:

```bash
bash experiments/qwen35_27b_migration/run_all.sh /secure/path/qwenkey.txt
```

The three protocols are started as detached processes. Logs and PID files are
written under `experiments/qwen35_27b_migration/logs/`; results are appended to
the corresponding `results/qwen3.5-27b_*` directory. Closing the terminal does
not stop them. A server reboot still requires restarting the jobs.

To run one protocol manually:

```bash
python3 experiments/qwen35_27b_migration/run_qwen35_api.py \
  --protocol single_score \
  --key-file /secure/path/qwenkey.txt \
  --output-dir experiments/qwen35_27b_migration/results/qwen3.5-27b_single_score
```

Override the endpoint with `BASE_URL=...` or `--base-url ...` when using a
different compatible service. The default model is `qwen3.5-27b`.

## Check progress

```bash
ps -ef | grep run_qwen35_api.py
tail -f experiments/qwen35_27b_migration/logs/single_score.log
wc -l experiments/qwen35_27b_migration/results/qwen3.5-27b_*/*.jsonl
```

The current checkpoint snapshot was copied from the running experiment and is
not a final benchmark table. It is included so another server can continue
without losing completed records.
