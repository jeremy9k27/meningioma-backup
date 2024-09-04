# %% This took roughly 1h10min to run on Zeus last time...
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup
from LOOExperiment import LOOExperiment
import time
from datetime import datetime
import numpy as np
import joblib

setup()

output_folder = 'results/LOO_T1only-radiomics-smoothed8_fine_9-4-24'
lambdas = np.arange(0.06, 0.16, 0.01) # coarse = np.arange(0.1, 15, 1) # fine = np.arange(0.06, 0.16, 0.01)

tasks = ['Chr22q', 'MethylationSubgroup', 'Chr1p'] # ['Chr22q', 'MethylationSubgroup', 'Chr1p']

begin_time = time.time()
start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f'\n\nStarted classic_loo.py at: {start_time}\n\n')

for task in tasks:
    print(f'\nStarting {task}...')
    exp = LOOExperiment(
        prediction_task=task, 
        lambdas=lambdas,
        output_dir=output_folder,
        use_smote=True,
        feat_file="data/radiomics/features8_smoothed/features_wide.csv", # data/5b_processed_normalized_features/features8_smoothed_constrainedByBrainMask_wide.csv", # f"data/collage_sparse/windowsize-9_binsize-64_summary_22nansfilled_pruned.csv" # "data/combined_feats/5-15-24_radiomics_pruned-collage_features.csv"
        feat_select="T1"
    )

    if task == 'MethylationSubgroup':
        pmetric = 'Balanced Accuracy'
    else:
        pmetric = 'Balanced Accuracy'

    exp.par_loo_model(pmetric=pmetric)

    joblib.dump(exp, f'{output_folder}/{task}/exp.pkl')
    
end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
time_elapsed = time.time() - begin_time
hours, rem = divmod(time_elapsed, 3600)
minutes, seconds = divmod(rem, 60)
time_elapsed = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
print(f'\n\nDone at {end_time}!\nTime elapsed: {time_elapsed}\n\n')
