# v0.2.0

This release stages the reproducibility package for the revised IEEE Signal Processing Letters manuscript. It is a source-and-derived-results release, not a dataset or model distribution.

## Included

- Calibration-only trajectory detector and evaluation utilities.
- Frozen passive, modern IQA, and adapted forensic-baseline result summaries.
- Protocol, artifact, runtime, and hash metadata needed to audit the released results.
- A deterministic synthetic image generator for installation and smoke testing. The demo uses a fixed seed and does not require model weights or external data.
- Root-level `pyproject.toml` for editable installation.

## Deliberately excluded

- Manuscript source, compiled PDF, tracked-changes files, and submission materials.
- CelebA-HQ images, parsing masks, complete datasets, and author-only manifests.
- Third-party source trees, learned-model checkpoints, and credentials.

The released IQA results that use IQA-PyTorch remain conditional on the pinned external runtime (`pyiqa==0.1.13`) and locally available model cache. The synthetic demo is only a plumbing check and must not be treated as scientific evidence.

The Zenodo concept DOI is `10.5281/zenodo.22098907`. The version DOI for `v0.2.0` is intentionally absent until the user synchronizes the GitHub release through Zenodo.
