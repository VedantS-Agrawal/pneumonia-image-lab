# Pneumonia Image Lab

A reproducible **28×28 PneumoniaMNIST** experiment and an interactive website explaining a small convolutional neural network.

Select a test X-ray, run actual browser inference, inspect every convolution/ReLU/pooling map, reveal the dataset label, and explore errors and classification thresholds. No uploads. No clinical use.

## Results from this run

| Measure | Value |
|---|---:|
| Test ROC-AUC | 0.952 |
| Image-bootstrap 95% interval | 0.934–0.968 |
| Sensitivity at validation-selected threshold | 96.9% |
| Specificity at validation-selected threshold | 69.7% |
| Accuracy | 86.7% |
| False positives / false negatives | 71 / 12 |
| Parameters | 4,753 |

One seed, one architecture, one resolution. These are benchmark results, not external clinical validation. The high sensitivity does not erase the substantial false-positive count.

Repository: https://github.com/VedantS-Agrawal/pneumonia-image-lab

## Live website

- [Pneumonia Image Lab](https://vedants-agrawal.github.io/pneumonia-image-lab/)
- [CheXNet interactive paper guide](https://vedants-agrawal.github.io/pneumonia-image-lab/paper/)

GitHub Pages serves the committed `dist/` directory. Pushes to `main` run the numerical/data checks and deploy through `.github/workflows/pages.yml`.

The paper guide explores the results of [arXiv:2505.06646v1](https://arxiv.org/html/2505.06646v1), alongside interactive explanations of multi-label outputs, patient splitting, DenseNet connections, focal loss, thresholds, prevalence, calibration, and attribution. Reported numbers and synthetic teaching examples are labeled separately. This is a reading companion, not a reproduction of the authors' model. See `docs/paper-guide.md`.

## Run the existing website

```sh
python3 -m http.server 8766 --directory dist
```

Open http://localhost:8766. The committed website includes the trained weights, sample images, evaluation results, and ONNX model, so Python training dependencies are not needed to view it.

## Reproduce the model

Use Python 3.10 or newer. On the original machine, Python 3.12 was used on Apple silicon.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.train
python -m src.export_fixtures
node tests/inference.mjs
```

`--device cpu` forces CPU; the default selects MPS on a compatible Mac, otherwise CPU. For exact package versions from this environment, use `requirements-lock.txt`; it records an Apple-silicon environment and may need adaptation elsewhere. CPU/MPS numerical behavior may vary even with the same seed.

Training downloads approximately 4 MB through the official MedMNIST loader. It does not resize the supplied 28×28 images. Inspect `configs/experiment.json` before running; the first version deliberately uses only size 28 and seed 42.

## Learning walkthrough

1. `src/model.py`: image normalization, three convolution blocks, and a single output.
2. `src/train.py`: data loading, class weighting, checkpoint selection, threshold selection, evaluation, and export.
3. `dist/inference.js`: the same forward-pass math, implemented in JavaScript to expose intermediate maps.
4. `dist/app.js`: interactive layer inspector, gallery, and threshold exploration.
5. `dist/paper/`: interactive paper companion and transcribed result tables.
6. `docs/methods.md`: experimental decisions and limitations.

Dataset labels are normal/pneumonia. Training uses class-weighted binary cross-entropy, so the displayed sigmoid score is not a calibrated probability of disease. Test-set slider changes are exploratory and do not choose the published default threshold.

## Repository layout

```text
configs/                 Experiment settings
src/                     Training and model code
runs/                    Local checkpoints and logs (ignored)
data/                    Downloaded dataset (ignored)
tests/                   Python and JavaScript verification
dist/                   Standalone static website, models, and results
docs/                   Methods and step-by-step learning guide
```

No dataset archive or environment is committed. Selected dataset images and trained model artifacts are included to make the website self-contained. Images are CC BY 4.0; see `DATA_ATTRIBUTION.md`.

## References

- [MedMNIST](https://medmnist.com/) and [dataset definitions](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py)
- Yang et al., *MedMNIST v2*, Scientific Data (2023).
- Kermany et al., *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, Cell (2018).
- Interface inspiration: [Medical CNN Explainer](https://ml-health.roshchupkin.org/cnn_explainer.html) and [CNN Explainer](https://poloclub.github.io/cnn-explainer/). Original implementation and styling; no source copied.
