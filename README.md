# Active Probe-Response IQA

This release contains the calibration-only trajectory detector and disclosure-safe derived results for **Active Probe-Response Detection of Spatially Selective Image Degradation**. It does not contain the manuscript, PDF, raw face images, parsing masks, model weights, or credentials.

## Reproduction boundary

The experiments use CelebA-HQ, which must be obtained from its official provider under the applicable access terms. Raw images and masks are intentionally not redistributed. Authorized users should place local inputs according to their own manifest and run the analysis scripts with portable paths; the frozen result JSON and report in `results/` provide the released reference outputs.

The trajectory detector applies fixed JPEG and Gaussian-blur probe sweeps and evaluates gradient-orientation and block-boundary response features. The held-out report records the detector AUCs and bootstrap intervals. The release also contains `analysis/evaluate_passive_baselines.py` and aggregate results under `results/passive-baselines/`. These compare the detector with a DCT high-frequency-ratio adapter and a local-texture-statistics adapter on the same frozen split. They are transparent paper-inspired adapters, not exact reimplementations of the cited published training pipelines.

## Software

Python 3.9+ is recommended. Install the package in editable mode from the repository root with `pip install -e .` after moving `environments/pyproject.toml` to a local project configuration if desired. The implementation uses NumPy and Pillow; no GPU is required for the released detector.

## Verification

```text
python -m pytest
python analysis/evaluate_iqa_ssc_trajectory_detector.py --help
python analysis/evaluate_passive_baselines.py --help
```

`results/trajectory_detector.json` and `results/trajectory_detector.md` are the frozen outputs used for the manuscript. `results/figures/trajectory_detector.png` is a disclosure-safe rendering of the probe curves.

## Citation and license

Use the metadata in `CITATION.cff`. The code and derived outputs are released under the MIT License. The source dataset remains subject to its provider's terms and is not covered by this repository license.

Release version: `v0.1.3`.
