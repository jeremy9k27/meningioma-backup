#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Execute preprocessing scripts in succession
python 4a_parallel_n4_correction.py
python 5a_skullstrip.py
python 6a_zscore_normalize.py
python 7a_parallel_registration_w_propogation.py

echo "All scripts executed successfully."
