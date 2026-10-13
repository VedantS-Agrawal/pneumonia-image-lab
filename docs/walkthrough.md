# Build and understand the project

## 1. Inspect the dataset

Read class counts and show sample images from each split. Check what the labels mean and why class imbalance can make accuracy misleading. Data lives in the ignored `data/` directory.

## 2. Read the model

Open `src/model.py`. Follow one tensor through the shapes:

`1×28×28 → 8×28×28 → 8×14×14 → 16×14×14 → 16×7×7 → 24×7×7 → 24 means → 1 logit`.

A sigmoid produces a 0–1 score. Ask why one output fits a binary task and why a weighted loss changes score interpretation.

## 3. Read the training loop

Open `src/train.py`. Find training loss, optimizer updates, validation evaluation, best-checkpoint selection, and early stopping. Identify where validation is used and verify the test set does not update model weights or select settings.

## 4. Reproduce

Follow README setup and training commands. Compare your run with the recorded result; device/software differences can change floating-point behavior. Keep changed settings in a new configuration and report the changes.

## 5. Interpret results

Inspect the 71 false positives and 12 false negatives at the chosen threshold. Ask what changing a threshold changes and what it does not. Distinguish ROC-AUC from accuracy and sensitivity from precision.

## 6. Inspect the website

Run the static server. Select an X-ray and reveal its label. Click a convolution map; choose an input channel and pixel. Compare the input patch and kernel with the resulting value. Try ReLU, max pooling, and output contributions.

## 7. Verify

Run `python -m unittest discover -s tests -p 'test_*.py'` and `node tests/inference.mjs`. Check the exported browser model agrees with Python before trusting the visualization. A visually plausible feature map alone is not verification.

## 8. GitHub and publishing

The static website is `dist/`. Training data/environments/checkpoints are ignored. The GitHub repository stores source, reproducibility documents, and small browser artifacts. Deployment can serve `dist/` with any static host; the included Sites manifest uses owner-private hosting unless sharing is changed explicitly.

## Future work

Only after this baseline is understood: repeat seeds, compare resolutions, compare a larger model, examine calibration, and assess external generalization. Do not present these future experiments as completed work.
