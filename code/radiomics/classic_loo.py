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

output_folder = 'data/classic_loo_pca_regression_5-1-24' # 'data/classic_loo_pca_regression_5-1-24' # 'data/classic_loo'
lambdas = np.arange(0.2, 2.2, 0.2) # np.arange(0.2, 2.2, 0.2) # coarse = np.arange(1.0, 9.0, 1.0) # fine = np.arange(0.2, 2.2, 0.2) # np.arange(0.06, 0.16, 0.02)

tasks = ['Chr22q'] # ['Chr22q', 'MethylationSubgroup', 'Chr1p']

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
        feat_file=f"data/pca_results/{task}/pca_scores.csv"
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
