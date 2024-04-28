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

output_folder = 'data/lto_experiment'
# lambdas = np.arange(0.01, 1.29, 0.01) # 128 different lambdas
lambdas = np.arange(0.05, 0.23, 0.01) # 18 different lambdas
tasks = ['Chr22q', 'MethylationSubgroup']
sanity_lambdas = {}
sanity_lambdas[tasks[0]] = [0.06, 0.08, 0.15, 0.18]
sanity_lambdas[tasks[1]] = [0.09, 0.12]

begin_time = time.time()
start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f'\n\nStarted lto_experiment.py at: {start_time}\n\n')

for task in tasks:
    print(f'\nStarting {task}...')
    for l in sanity_lambdas[task]:
        exp = LTOExperiment(
            prediction_task=task, 
            lambdas=lambdas,
            output_dir=output_folder,
            use_smote=True,
            debug=False,
            best_lambda=l
        )

        exp.final_model_loo(max_workers=32)

        # joblib.dump(exp, f'{output_folder}/{task}/exp.pkl')
    
end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
time_elapsed = time.time() - begin_time
hours, rem = divmod(time_elapsed, 3600)
minutes, seconds = divmod(rem, 60)
time_elapsed = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
print(f'\n\nDone at {end_time}!\nTime elapsed: {time_elapsed}\n\n')
