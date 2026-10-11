#!/bin/bash
# Runs all plot-level experiments (pre-registered design). Resumable.
set -e
cd "$(dirname "$0")/.."
python3 -m src.experiments.run_plot_experiments --exp harvest_ltae    --model ltae    --variants A B C D E F
python3 -m src.experiments.run_plot_experiments --exp harvest_gridxgb --model gridxgb --variants A B C D E F
python3 -m src.experiments.run_plot_experiments --exp reduced_ltae    --model ltae    --variants B C --fracs 0.1 0.25 0.5
python3 -m src.experiments.run_plot_experiments --exp yield_ltae      --model ltae    --target YIELD --variants A B C D E
