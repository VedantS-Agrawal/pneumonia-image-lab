# Methods and limitations

## Scope

First version: one small CNN trained on the official 28×28 PneumoniaMNIST images. No additional resizing, resolution comparison, augmentation, or transfer learning. Future experiments should be pre-specified and kept separate from this baseline.

## Data

- Official splits: 4,708 train, 524 validation, 624 test.
- Class counts: train 1,214 normal / 3,494 pneumonia; validation 135 / 389; test 234 / 390.
- uint8 grayscale images -> float32 / 127.5 - 1.
- These are downsampled pediatric images and image-level labels. No DICOM metadata, reports, segmentation annotations, or patient IDs are used in this experiment.
- Images are downloaded using MedMNIST's built-in URL and checksum verification.

## Model and training

- 3×3 convolutions: 1→8→16→24 channels, padding 1.
- ReLU after each convolution; 2×2 max pooling after the first two.
- Global spatial average produces 24 features; linear output produces one logit.
- 4,753 trainable parameters.
- AdamW: learning rate 0.002, weight decay 0.0001, batch size 64.
- Positive loss weight = training normal count / pneumonia count. No test information enters this choice.
- Up to 30 epochs; patience 8 on validation ROC-AUC. Seed 42.
- Checkpoint selected on validation ROC-AUC. Actual selected epoch: 29.
- Threshold maximizes validation balanced accuracy (Youden J): 0.5766407251. Weighted training and threshold selection mean scores should not be interpreted as calibrated disease probabilities.

## Evaluation

Only the finalized checkpoint/threshold is applied to the test set. ROC-AUC, average precision, accuracy, balanced accuracy, sensitivity, specificity, and counts are saved in `dist/assets/results.json`. The threshold slider is an exploratory visualization of the fixed test scores, not a tuning tool.

The ROC-AUC interval uses 1,000 image-level bootstrap samples, seed 2026. It does not account for patient clustering, training-seed variation, or external population shift. One run cannot establish robustness across random seeds.

Gallery selection: 24 random test indices (seed 17) plus up to eight errors farthest from the selected threshold. This deliberately enriches mistakes; gallery frequency must not be used to estimate model performance. All displayed aggregate metrics use all 624 test images.

## Export and verification

- Native `.pt` checkpoints remain in ignored `runs/`.
- JSON exports contain trained weights. Browser inference uses those weights and identical normalization.
- ONNX exports are checked against PyTorch on 16 test images: maximum logit difference approximately 4.77e-6 in this run.
- JavaScript inference is checked against Python predictions on 10 fixtures, including all relevant operations. Maximum score difference approximately 6.21e-8.
- The operation inspector shows actual channel patches, learned kernels, bias, and final values. Displayed values are rounded; calculations use full precision.
- Feature maps are activations, not localized disease annotations or proof of clinical reasoning.

## Timing

The recorded MPS training duration is about 19.5 seconds for this run, excluding environment installation and dataset download. This is a tiny model on low-resolution images, not an estimate for larger models. Single-image CPU timing is a warm median over 50 calls with four PyTorch threads. Browser timing measures the forward pass and excludes image/model loading; it is not directly comparable with a different device's training or CPU timing.

## Intended use

Educational portfolio and reproducibility demonstration only. No clinical validation, calibration claim, adult generalization, or approved medical use. Repeated future access to test results can invalidate the interpretation of an untouched final test; new hypotheses should be evaluated appropriately and reported transparently.
