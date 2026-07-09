# VisPrefDiag

This directory is the cleaned project copy for the paper:

**Not a Perception Problem: Visual Preference Evaluation Fails Where Signals Vanish**

## Layout

- `benchmark/`: benchmark metadata needed by the diagnostic scripts.
- `method/`: evaluator and VLM-judge method code copied from `Valuator`.
- `show/experiment/`: runnable or traceable experiment bundles, named with the requested `[experiment n]...` and `[phenomenon n]...` convention.
- `show/figure/`: final paper figure assets only.
- `show/tabel/`: final paper table assets only. The folder keeps the requested spelling `tabel`.

## Copy Policy

The source `Valuator` directory is left unchanged. This copy excludes raw ImageReward `.webp` image dumps, large zip archives, `.git`, `__pycache__`, `.log`, `.pid`, `.lock`, `.DS_Store`, and other runtime noise. Raw ImageReward images remain in `../Valuator/benchmark/ImageReward/` if needed for full reruns.

## Paper Mapping

- Main Hard/Mid/Easy diagnostic: `show/experiment/[experiment 1]ImageReward main difficulty split`
- Interface perturbations: `show/experiment/[experiment 2]interface perturbation robustness`
- Pick-a-Pic validation: `show/experiment/[experiment 3]Pick-a-Pic generalization`
- Temperature ablation: `show/experiment/[experiment 4]temperature ablation`
- Hardness protocol ablation: `show/experiment/[experiment 5]hardness protocol ablation`
- Hard-instance gap assets: `show/experiment/[phenomenon 1]hard-instance gap`
- Zero-gap collapse assets: `show/experiment/[phenomenon 2]zero-gap signal collapse`
- Protocol separation figure assets: `show/experiment/[phenomenon 3]protocol separation`
