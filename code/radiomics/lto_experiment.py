# %%
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup
from LTOExperiment import LTOExperiment
import time
from datetime import datetime
import numpy as np
import joblib

setup()

output_folder = 'data/lto_experiment_debug'
# lambdas = np.arange(0.01, 1.29, 0.01) # 128 different lambdas
lambdas = [0.08, 0.12]
# tasks = ['Chr22q', 'MethylationSubgroup', 'Chr1p']
tasks = ['Chr22q', 'MethylationSubgroup']

begin_time = time.time()
start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f'\n\nStarted classic_loo.py at: {start_time}\n\n')

for task in tasks:
    print(f'\nStarting {task}...')
    exp = LTOExperiment(
        prediction_task=task, 
        lambdas=lambdas,
        output_dir=output_folder,
        use_smote=True
    )

    if task == 'MethylationSubgroup':
        pmetric = 'Macro AUC'
    else:
        pmetric = 'AUC'

    exp.par_lto_model(pmetric=pmetric)

    joblib.dump(exp, f'{output_folder}/{task}/exp.pkl')
    
end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
time_elapsed = time.time() - begin_time
hours, rem = divmod(time_elapsed, 3600)
minutes, seconds = divmod(rem, 60)
time_elapsed = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
print(f'\n\nDone at {end_time}!\nTime elapsed: {time_elapsed}\n\n')
