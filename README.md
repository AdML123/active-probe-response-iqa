# Active Probe-Response IQA

This `v0.2.0` release contains the calibration-only trajectory detector, disclosure-safe derived results, strong-baseline adapters, and a weight-free synthetic smoke test for **Active Probe-Response Detection of Spatially Selective Image Degradation**. It does not contain the manuscript, PDF, raw face images, parsing masks, model weights, or credentials.

## Reproduction boundary

The experiments use CelebA-HQ, which must be obtained from its official provider under the applicable access terms. Raw images and masks are intentionally not redistributed. Authorized users should place local inputs according to their own manifest and run the analysis scripts with portable paths; the frozen result JSON and report in `results/` provide the released reference outputs.

The trajectory detector applies fixed JPEG and Gaussian-blur probe sweeps and evaluates gradient-orientation and block-boundary response features. The held-out report records the detector AUCs and bootstrap intervals. The release also contains `analysis/evaluate_passive_baselines.py` and aggregate results under `results/passive-baselines/`. These compare the detector with a DCT high-frequency-ratio adapter and a local-texture-statistics adapter on the same frozen split. They are transparent paper-inspired adapters, not exact reimplementations of the cited published training pipelines.

`analysis/evaluate_pyiqa_baselines.py` runs the same split through the external IQA-PyTorch backend. The released classical audit (`results/strong-baselines/pyiqa-audit-classical.json`) contains BRISQUE, NIQE, and PIQE row-level AUCs. The learned panel is complete in `results/strong-baselines/pyiqa-audit-learned-summary.json`, with one per-model JSON report. MUSIQ, TOPIQ-NR, ARNIQA, LIQE, and CLIP-IQA use the frozen pyiqa configurations; MANIQA uses an explicitly recorded `test_sample=1` compute-bounded adaptation. These are held-out row-level AUCs, not energy-matched pair reversal rates.

`analysis/evaluate_pyiqa_pair_audit.py` computes the separate energy-matched JPEG pair audit. It reads a locally supplied frozen pair file, verifies its SHA-256 digest, regenerates each declared bilateral and JPEG member, orients each score so that larger values mean better quality, and records whether the bilateral member is ranked above its matched JPEG control. The released JSONL contains disclosure-safe scores and pair identifiers; it does not contain source images. Pair reversal and row-level AUC answer different questions and must not be substituted for one another.

Before scoring, the pair-audit evaluator reads `pyiqa.__version__` from the imported runtime and checks it against both `artifact-lock.json` and `modern-pair-audit-lock.json`. It also verifies the artifact-lock digest, the evaluator/source/transform digests, and a canonical fingerprint of the frozen metric and bootstrap configuration. A mismatch stops the run before model construction; the backend label in generated rows and summaries is derived from the verified runtime rather than a hard-coded version string.

`analysis/evaluate_forensic_adapters.py` evaluates the disclosed Ding 2019 and Shehin 2022 feature adaptations. Their full held-out report is `results/strong-baselines/forensic-adapters-summary.json`. Ding fixes a 5x5 vertical edge-patch geometry; Shehin applies the published frequency-ratio construction to the protocol's fixed OpenCV bilateral operator instead of an adaptive bilateral filter. Both deviations are recorded in the result metadata, and neither adapter is presented as an exact source-pipeline reproduction.

## Software

Python 3.9+ is recommended. The frozen detector uses NumPy, OpenCV, and PyWavelets; no GPU is required. The root `pyproject.toml` is the install contract; `environments/pyproject.toml` records the same base dependencies for environment provisioning. The checked-in `pytest.ini` applies the `src/` layout for the test suite.

Modern IQA candidates are registered in `results/strong-baselines/artifact-lock.json`. They require an external, pinned IQA-PyTorch runtime and its model cache. IQA-PyTorch source and weights are not redistributed here, and no learned-baseline result is claimed unless the backend conformance run succeeds.

The package itself is installable from the repository root. The validated reference runtime is the local conda environment `paper20-cu128` with Python 3.10.20, `pyiqa==0.1.13`, PyTorch `2.11.0+cu128`, CUDA 12.8, and scikit-learn 1.7.2. Install the package and pinned optional IQA backend in that environment:

```text
conda run -n paper20-cu128 python -m pip install -e ".[iqa]"
```

The fixed 20-image conformance panel still requires a local image root and manifests, which are not part of this release:

```text
conda run -n paper20-cu128 python -m pip install pyiqa==0.1.13
conda run -n paper20-cu128 python analysis/run_learned_iqa.py --image-root <LOCAL_CELEBA_ROOT> --manifest <LOCAL_CALIBRATION_MANIFEST> --output results/strong-baselines/pyiqa-conformance-20.json --device cuda --limit 20 --batch-size 1
conda run -n paper20-cu128 python analysis/evaluate_pyiqa_baselines.py --calibration-manifest <LOCAL_CALIBRATION_MANIFEST> --evaluation-manifest <LOCAL_EVALUATION_MANIFEST> --image-root <LOCAL_CELEBA_ROOT> --summary-out results/strong-baselines/pyiqa-audit-learned-summary.json --device cuda --bootstrap 2000 --metrics musiq maniqa topiq_nr liqe arniqa clipiqa --batch-size 4 --maniqa-test-sample 1
conda run -n paper20-cu128 python analysis/evaluate_pyiqa_pair_audit.py --pair-json <LOCAL_FROZEN_JPEG_PAIR_JSON> --image-manifest <LOCAL_FROZEN_IMAGE_MANIFEST> --image-root <LOCAL_CELEBA_ROOT> --lock-json results/strong-baselines/modern-pair-audit-lock.json --output-dir results/strong-baselines --device cuda --batch-size 4 --maniqa-test-sample 1 --bootstrap 2000 --metrics musiq maniqa topiq_nr arniqa liqe clipiqa
```

The adapter applies a local compatibility shim for the legacy CLIP import used by LIQE and CLIP-IQA and scores high-resolution images in bounded batches. `results/strong-baselines/conformance-summary-paper20.json` records the environment and explains why this panel is a reproducibility check, not a manuscript evaluation result.

Because this pyiqa release pins `transformers==4.37.2`, the reference environment may report a conflict with unrelated `sentence-transformers` installations. Keep it isolated for IQA reproduction.

## Verification

The following command creates two deterministic PNG inputs and a manifest under a temporary directory. It uses no external dataset, model weights, or network access:

```text
python scripts/generate_synthetic_demo.py --output-dir demo-output
python analysis/run_iqa_ssc_trajectory_detector.py --image-root demo-output --manifest demo-output/manifest.json --output demo-output/trajectories.jsonl --count 2
```

The generated directory is disposable and is not included in the release archive. It is intended to confirm installation and image-processing plumbing, not to reproduce manuscript metrics.

```text
python -m pytest
python analysis/evaluate_iqa_ssc_trajectory_detector.py --help
python analysis/evaluate_passive_baselines.py --help
```

`results/trajectory_detector.json` and `results/trajectory_detector.md` are the frozen outputs used for the manuscript. `results/figures/trajectory_detector.png` is a disclosure-safe rendering of the probe curves.

## Citation and license

Use the metadata in `CITATION.cff`. The code and derived outputs are released under the MIT License. The source dataset remains subject to its provider's terms and is not covered by this repository license. The Zenodo concept DOI for this release line is [10.5281/zenodo.22098907](https://doi.org/10.5281/zenodo.22098907); a version DOI is added only after the user archives `v0.2.0`.

Release version: `v0.2.0`.
