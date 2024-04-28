# %%
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup
from DebugEnsemble import LTOExperiment
import time
from datetime import datetime
import joblib

setup()

output_folder = 'data/debug_ensemble'
tasks = ['Chr22q', 'Chr1p'] # 'MethylationSubgroup'
lambdas = [1.0, 0.9]
begin_time = time.time()
start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f'\n\nStarted debug_ensemble_exp.py at: {start_time}\n\n')

for i, task in enumerate(tasks):
    print(f'\nStarting {task}...')
    exp = LTOExperiment(
        prediction_task=task, 
        lambdas=lambdas,
        output_dir=output_folder,
        use_smote=False,
        debug=False,
        best_lambda=lambdas[i]
    )

    if task == 'MethylationSubgroup':
        pmetric = 'Macro AUC'
    else:
        pmetric = 'AUC'

    exp.process_lambda(lambdas[i])

    joblib.dump(exp, f'{output_folder}/{task}/exp.pkl')
    
end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
time_elapsed = time.time() - begin_time
hours, rem = divmod(time_elapsed, 3600)
minutes, seconds = divmod(rem, 60)
time_elapsed = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
print(f'\n\nDone at {end_time}!\nTime elapsed: {time_elapsed}\n\n')
