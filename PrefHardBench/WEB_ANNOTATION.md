# PrefHardBench Web Annotation

This is a dependency-free local annotation interface. It reads the existing
`data_annotations.csv` and images, but does not expose the existing `judge`,
`confidence`, or `reason` fields to annotators. New responses are appended to a
JSONL file and can be merged with the benchmark later.

## Start

From the `AUCTION` directory:

```bash
python3 PrefHardBench/web_app.py --port 8765
```

Open <http://127.0.0.1:8765>. To run a small pilot first:

```bash
python3 PrefHardBench/web_app.py --port 8765 --limit 20 \
  --output PrefHardBench/annotations/pilot.jsonl
```

For the actual collection, use a preselected stratified manifest instead of the
first rows in the CSV:

```bash
python3 PrefHardBench/web_app.py --port 8765 \
  --question-list PrefHardBench/annotation_manifest.txt \
  --output PrefHardBench/annotations/human_round1.jsonl
```

The manifest is either one `question_id` per line or a CSV with a
`question_id` column.

The default output is `PrefHardBench/annotations/prefhard_annotations.jsonl`.
The server creates this file when the first response is submitted.

Aggregate repeated ratings after collection:

```bash
python3 PrefHardBench/aggregate_annotations.py \
  --input PrefHardBench/annotations/prefhard_annotations.jsonl \
  --output PrefHardBench/annotations/prefhard_aggregated.csv
```

The aggregate file contains per-question annotation count, mean/median/std of
difficulty, confidence mean, A/B/U counts, majority preference, majority
fraction, and reason counts.

## Collected fields

Each response stores the anonymous annotator ID, question ID, original and
displayed image order, A/B/U preference, direct 1--5 difficulty, 1--5 response
confidence, conditional attribution code, optional note, and elapsed time.
The same anonymous ID cannot submit the same question twice, while different
anonymous IDs can independently annotate the same question.

The existing labels are used only to construct the question pool and are never
sent to the browser. The app swaps image order deterministically by annotator ID
and question ID, and records the order so the result can be mapped back to the
original image identities.

## Collection recommendation

For the first human-grounded PrefHardBench release, select 300--500 questions
and obtain 3--5 independent annotations per question. Keep the direct difficulty
rating for every A/B/U response. Split later experiments by prompt or question
group, not by individual rows.
