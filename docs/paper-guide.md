# Interactive paper guide

Source: [Strick, Garcia, Huang — Reproducing and Improving CheXNet, arXiv:2505.06646v1](https://arxiv.org/html/2505.06646v1), posted May 10, 2025. Pin the version when discussing the displayed results; the authors' current code repository may have evolved.

## What the page contains

- **Reported:** values transcribed from Tables 2–4 and a concise description of the study's task and training setup.
- **Simulated:** label vectors, patient splitting, network connections, brightness adjustment, score thresholds, class thresholds, prevalence, attribution patterns, and calibration examples.
- **Interpretation:** questions about fair threshold comparisons, objective differences, uncertainty, and generalization.

No study weights or raw predictions are included. No synthetic ROC curve, confusion matrix, heatmap, or calibration measurement is represented as an empirical result from the paper. The small lab's image used for brightness illustration is explicitly separate from NIH ChestX-ray14.

## Transcription and comparison rules

- Preserve source precision; do not derive new headline values by averaging rounded disease rows.
- Table 4 omits pneumothorax. Its F1 stays missing; original CheXNet F1 is not supplied for these comparisons.
- Keep Table 3 aggregate values exactly as reported, even when rounded disease rows do not reproduce them.
- DannyNet uses class-specific thresholds; the two baselines use 0.5. Their F1 comparison mixes model and threshold-policy differences.
- Focal-loss and BCE values do not share an objective. Lower displayed loss alone does not demonstrate improved calibration.
- Original-study AUC differences are descriptive cross-study comparisons, not a controlled trial.
- The 28×28 binary pediatric lab and the paper's 14-label task cannot be directly ranked by their AUC values.

## Verify and edit

`dist/paper/data.js` holds table values and deterministic synthetic examples. `tests/paper.mjs` checks the data contracts, missing values, AUC tie handling, focal-loss identity, and threshold behavior. `dist/paper/paper.js` handles the interactions; `paper.css` extends the lab's visual style.

Run `node tests/paper.mjs`, preview `dist/`, then check desktop/mobile layout and controls before pushing. The workflow publishes both pages together to GitHub Pages.
