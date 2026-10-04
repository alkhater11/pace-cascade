# PACE: input-adaptive cascaded approximate computing (reproducibility package)

Code, configurations, energy-model parameters, seeds and result files for the article

> **Energy-Efficient AI Acceleration at the Edge: An Input-Adaptive Cascaded
> Approximate-Computing Framework for Large-Scale Machine Learning**
> Journal of King Saud University – Computer and Information Sciences.

The study asks whether running a cheap, aggressively approximated configuration
first and escalating only low-confidence inputs to a higher-precision
configuration (an *input-adaptive precision cascade*) can beat the best **static**
mixed-precision configuration, found by exhaustive enumeration. Everything that
produces a number, table or figure in the paper is in this repository, and every
run is deterministic.

## Quick start

```bash
git clone https://github.com/USERNAME/pace-cascade.git && cd pace-cascade
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/download_data.py      # MNIST, Fashion-MNIST, CIFAR-10 mirror (checksum-verified)
python3 tests/test_pace.py            # unit tests, about 1 minute
bash reproduce_all.sh                 # everything, about 5-6 hours on ONE CPU core
```

`reproduce_all.sh` lists every stage with its approximate run time; stages can be
run individually from `code/` (each script reads its settings from `configs/`).

## Environment used for the paper

Single x86-64 CPU core, 3 GB RAM, no GPU. Python 3.12, NumPy 2.4.4, SciPy 1.17.1,
Matplotlib 3.10.8, Pillow 12.1.1, JAX 0.11.2 (CPU; used only to *train* ResNet-8
and ViT-4). All approximate inference runs in NumPy.

## Determinism and exactness

* Every run is seeded (initialisation, shuffling, augmentation, corruptions).
  Re-training the MLPs with `run_train.py` reproduces the archived accuracies of
  the first revision exactly (checked in `run_r2_mlp.py`, section "reproduction").
* Approximate arithmetic is **simulated exactly**: operands are quantized to
  integers, the `t` least-significant bits of both operands are zeroed (operand
  truncation), and the multiply-accumulate is computed on integer-valued float64
  numbers whose sums stay far below 2^53, so the result is bit-exact. No random
  error injection is used anywhere. `tests/test_pace.py` checks this, and checks
  that the NumPy inference engines reproduce the JAX-trained models.

## Repository layout

```
configs/                 one JSON file per experiment + energy_models.json
code/
  axlib.py               data loading, NumPy MLP, quantizer + truncated multiplier, 45 nm energy model
  cnnlib.py              NumPy SmallCNN (R1 generality study)
  deepnets.py            JAX ResNet-8 (CIFAR-10) and ViT-4 (Fashion-MNIST): definitions + training
  deepquant.py           integer-exact approximate inference for ResNet-8 / ViT-4, prefix-cached enumeration
  r2lib.py               energy-model families, exact cascade curves, envelopes, Proposition 3 quantities,
                         deployment protocol, corruptions
  run_*.py               experiments (see the mapping table below)
  make_figs*.py, mstyle.py   figures (MATLAB-like style), PDF + EPS
scripts/download_data.py dataset download with MD5 verification
tests/test_pace.py       unit tests
results/                 JSON/CSV result files behind every table and figure
figs/                    generated figures
reproduce_all.sh         full pipeline
```

## Configuration files (`configs/`)

| file | content |
|---|---|
| `mlp.json` | MLP 784-256-128-10: seeds, optimiser, epochs, batch size, split (note: **no weight decay**) |
| `mlp_narrow.json` | narrow MLP 784-32-10 used as the cheap arm of the capacity cascade |
| `cnn_small.json` | SmallCNN of the first revision (architecture, 20k training subset, 81 configurations ranked on 1,500 validation images) |
| `resnet8_cifar10.json` | ResNet-8: split, SGD schedule, augmentation, seeds |
| `vit4_fmnist.json` | ViT-4: patching, width, depth, heads, AdamW schedule, seeds |
| `energy_models.json` | per-MAC energy models H0-H6 and the (g, phi) family grid |

Option sets, corruption types and severities are defined once in `code/r2lib.py`
(`OPTIONS`, `OPTIONS_EXT`, `DEEP_OPTIONS`, `CORRUPTIONS`, `SEVERITY`).

## Energy models (`configs/energy_models.json`)

Per-MAC energy as a function of the operand width `b` and the effective
(post-truncation) width `beta = b - t`:

| model | architecture class | per-MAC energy (pJ) |
|---|---|---|
| H0 | bit-parallel array MAC, fixed 32-bit accumulator, 45 nm (main text; Horowitz 2014) | 0.1 + 0.003125·beta² |
| H1 | array MAC with precision-scaled accumulator | 0.03125 + 0.00625·beta + 0.003125·beta² |
| H2 | bit-serial datapath | 0.0375·beta |
| H3 | H0 + operand fetch (first revision) | 0.1 + 0.003125·beta² + 0.15·b |
| H4 | voltage-accuracy-scaled array (DVAFS-like) | (0.1 + 0.003125·beta²)·V(beta)², V = 0.6 + 0.05·beta |
| H5 | analog in-memory MAC, converter-dominated | 0.03 + 0.00105469·2^beta |
| H6 | fixed-width INT8 SIMD lanes | 0.3 |
| family | e8·[phi + (1−phi)·g(beta)/g(8)], g ∈ {beta, beta², beta³, 2^beta}, phi ∈ {0, .05, .1, .2, .333, .5} | |

FP32 MAC: 4.6 pJ (3.7 pJ multiply + 0.9 pJ add). Conclusions depend only on
energy *ratios* (Proposition 4 of the paper), so the 45 nm scale is immaterial.

## Paper item → script → result file

| paper item | script(s) | result file(s) |
|---|---|---|
| Static design space, Pareto frontier, per-layer sensitivity (Sec. 7.2–7.3) | `run_train.py`, `run_explore.py` | `explore_{mnist,fmnist}.json`, `fp_summary.json` |
| Cascade vs. frontier, gates, calibration, oracle, headline table (Sec. 7.4–7.9) | `run_pace.py`, `run_perclass.py` | `pace_*.json`, `perclass.json`, `table_*.csv` |
| 36-pair sweep, per-seed selection (Sec. 8.1–8.2) | `run_revision.py`, `analyze_multipair.py` | `revision_r4.json`, `multipair_env_*.json` |
| SmallCNN (Sec. 8.3) | `run_cnn_train.py`, `run_cnn_pace.py` | `cnn_results.json`, `cnn_fp_summary.json` |
| Memory-augmented energy model (Sec. 8.4) | `run_energy_sensitivity.py`, `run_energy_fair.py` | `energy_fair.json`, `energy_sensitivity.json` |
| ResNet-8 / CIFAR-10 and ViT-4 / Fashion-MNIST (Sec. 8.5) | `deepnets.py`, `run_r2_deep.py` | `r2_resnet8.json`, `r2_vit4.json`, `*_fp_summary.json` |
| Corrupted and mixed-difficulty inputs (Sec. 8.6) | `run_r2_mlp.py`, `run_r2_deep.py` | `r2_mlp.json` (`corruption*`), `r2_resnet8.json`, `r2_vit4.json` |
| Predicted vs. observed, hardware regimes, precision gaps (Sec. 9.1–9.2) | `run_r2_regimes.py`, `run_r2_mlp.py` | `r2_regimes.json`, `r2_mlp.json` (`gap_*`) |
| Cascades between neighbouring configurations, time-sharing baseline (Sec. 9.3) | `run_r2_interp.py` | `r2_interp.json` |
| Width-and-precision (capacity) cascade (Sec. 9.4) | `run_capacity_train.py`, `run_r2_mlp.py` | `r2_mlp.json` (`capacity`), `narrow_fp_summary.json` |
| Figures 1–12 | `make_figs.py`, `make_figs_rev.py` | `figs/Fig1`–`Fig12` (.pdf, .eps) |
| Figures 13–17 | `make_figs_r2.py` | `figs/Fig13`–`Fig17` (.pdf, .eps) |

## Data provenance

See `data/README.md`. MNIST and Fashion-MNIST are fetched from GitHub mirrors of
the original IDX files and verified against the official MD5 checksums. The
canonical CIFAR-10 host was not reachable from the build environment, so the
experiments used the public mirror `YoongiKim/CIFAR-10-images`, which stores the
images as JPEG (quality 75); absolute CIFAR-10 accuracies are therefore not
directly comparable with published numbers, while all methods are compared on
identical data.

## Correction note

The first-revision manuscript stated a weight decay of 1e-4 for the MLPs; the
released code (and therefore every reported MLP result) uses **no weight decay**.
The second revision of the manuscript states this correctly.

## License and citation

MIT License (see `LICENSE`). Please cite the article (see `CITATION.cff`).
