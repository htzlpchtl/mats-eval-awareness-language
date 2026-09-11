# Figures used in the report

The table maps report numbers to filenames, whose numerical prefixes reflect
creation order rather than report order.

| Report figure | Output | Generator in src/ |
| --- | --- | --- |
| 1: Dataset design | presentation/05_dataset_design.png and .svg | make_presentation_figures.py; dataset_design_diagram.py |
| 2: Probe training and results | presentation/07_probe_training_and_results.png and .svg | probe_comparison_diagram.py |
| 3: Full prompt examples | presentation/01_dataset_examples.png and .svg | make_presentation_figures.py |
| 4: Length distributions | ../metrics/report_followup/length_distributions.png | report_followup_diagnostics.py |
| 5: Expanded layer curve | presentation/08_probe_b_layers_expanded.png and .svg | expanded_layer_figure.py |
| 6: Baselines and matching | presentation/04_baselines_and_matching.png and .svg | make_presentation_figures.py |
| 7: Source overlap | presentation/06_source_overlap.png and .svg | make_presentation_figures.py |
| 8: Translation scores | presentation/03_translation_scores.png and .svg | make_presentation_figures.py |

Output paths above are relative to results/figures/. The earlier combined
02_english_layers_and_baselines figure is retained as an alternative.

## Reproduce

From the existing repository root, with numpy, pandas, matplotlib and
scikit-learn installed, run in this order:

```bash
python -m src.make_presentation_figures
python -m src.probe_comparison_diagram
python -m src.expanded_layer_figure
python -m src.report_followup_diagnostics
```

These commands read the repository's saved selected data, CV metrics and
predictions. They require no activation files, GPU inference, translation API
calls or probe refitting. Run the main figure script first because the
follow-up diagnostics read its generated source-link CSV.

## Interpretation and provenance

- Layer curves show mean training-CV AUROC; shading is the fold range, not a
  confidence interval. Each block has a separate fitted probe.
- Baseline circles indicate held-out test results; diamonds indicate training
  CV. These are different evaluation partitions. Unmatched samples still obey
  length bounds; they overlap and are not independent replications.
- Translation uses the same 360 prompts and frozen English Probe B. Score zero
  is the English threshold. AUROC is calculated from scores, not predicted labels.
- Source-link arrows mean rewritten version to original version: 19 train/train,
  7 train/test, 14 test/train and 3 test/test. Shared sources do not imply exact
  duplicate text or quantify performance inflation.
- Dataset examples preserve source wording, capitalisation and punctuation;
  they illustrate the groups rather than feature prevalence. Their exact IDs,
  texts and selection metadata are recorded in figure_data.json.
- figure_data.json also records plotted metrics and input hashes.
  06_shared_source_links.csv contains the original/rewrite relationships.
- results/metrics/report_followup/ contains exploratory prompt-characteristic
  counts and the removal-of-related-test-prompts check. It does not contain
  results from a corrected split and refit.
