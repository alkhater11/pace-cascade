#!/bin/bash
# Reproduce every number, table and figure of the paper (single CPU core).
# Approximate wall-clock times on one x86-64 core are given per stage.
set -e
cd "$(dirname "$0")"
python3 scripts/download_data.py                       # ~1 min
python3 tests/test_pace.py                             # ~1 min
cd code
# --- main study and first-revision experiments (MLP, SmallCNN) ---
python3 run_train.py                                   # ~4 min   5 seeds x 2 datasets
python3 run_explore.py                                 # ~2 min   216-config enumeration, Pareto front
python3 run_pace.py                                    # ~2 min   cascade, gates, calibration, oracle
python3 run_perclass.py                                # <1 min
python3 run_revision.py && python3 analyze_multipair.py   # ~3 min  36-pair sweep, per-seed selection
python3 run_cnn_train.py                               # ~1.5 h  SmallCNN, 3 seeds x 2 datasets (NumPy)
python3 run_cnn_pace.py                                # ~15 min
python3 run_energy_sensitivity.py && python3 run_energy_fair.py   # ~2 min
# --- second-revision experiments ---
python3 run_capacity_train.py                          # ~2 min   narrow 784-32-10 MLPs
python3 run_r2_mlp.py                                  # ~15 min  precision gaps, corruptions, capacity cascade
python3 deepnets.py --model vit4                       # ~13 min  ViT-4, 3 seeds x 8 epochs (JAX)
python3 deepnets.py --model resnet8                    # ~25 min  ResNet-8, 3 seeds x 4 epochs (JAX)
python3 run_r2_deep.py --model vit4                    # ~10 min
python3 run_r2_deep.py --model resnet8                 # ~12 min
python3 run_r2_regimes.py                              # ~5 min   hardware regimes, predicted vs observed
python3 run_r2_interp.py                               # ~3 min   neighbouring-configuration cascades, time-sharing baseline
# --- figures ---
python3 make_figs.py && python3 make_figs_rev.py && python3 make_figs_r2.py
echo "done: see results/ and figs/"
