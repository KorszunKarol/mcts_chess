#!/home/karolito/miniconda3/envs/chess/bin/python
"""
Simple launcher for Coach Tal UCI engine.
This file uses the conda python directly in the shebang.
"""
import sys
import os

# Set working directory
os.chdir('/home/karolito/DL/chess_2.0')

# Add project to path
sys.path.insert(0, '/home/karolito/DL/chess_2.0')

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import and run the UCI engine
from coach_tal_uci import main
main()





