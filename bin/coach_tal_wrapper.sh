#!/bin/bash
# Wrapper script for Coach Tal UCI engine
# This ensures the conda environment is used correctly

cd /home/karolito/DL/chess_2.0
source /home/karolito/miniconda3/etc/profile.d/conda.sh
conda activate chess
exec /home/karolito/miniconda3/envs/chess/bin/python -u /home/karolito/DL/chess_2.0/coach_tal_uci.py "$@"





