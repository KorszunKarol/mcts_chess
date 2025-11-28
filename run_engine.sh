#!/bin/bash
# This script acts as a wrapper to launch the UCI engine.
# It activates the correct conda environment before running the python script
# and logs crash reports (stderr) to a separate file, letting stdout communicate with the GUI.

# Get the directory where this script is located to use absolute paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CRASH_LOG_FILE="$SCRIPT_DIR/engine_crash.log"

# Find the base of the conda installation
CONDA_BASE=$(conda info --base)
if [ -z "$CONDA_BASE" ]; then
    # If conda not found, log to the crash file and exit
    echo "FATAL: Conda installation not found." > "$CRASH_LOG_FILE"
    exit 1
fi

# Source the master conda script to make 'conda' command available
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate the target environment
conda activate tf

# Now, run the UCI engine script using the Python from the activated environment.
# stdout is NOT redirected, so it can communicate with xboard.
# stderr IS redirected, to capture any crash information.
exec "$(which python3)" "$SCRIPT_DIR/uci_engine.py" 2> "$CRASH_LOG_FILE"