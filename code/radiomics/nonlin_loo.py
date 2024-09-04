# %% This took roughly 1h10min to run on Zeus last time...
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup
from NonlinearLOOExp import NonlinearLOOExp
import time
from datetime import datetime
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct

setup()

output_folder = 'results/Nonlin_LOO_SVM_T1-post-radiomics8_9-4-24'
model = SVC
fixed_params = {
    'C': 1,
    'gamma': 'auto',
    'class_weight': 'balanced',
    'probability': True
}
param_to_sweep = 'kernel'
sweep_values = ['linear', 'poly', 'rbf', 'sigmoid']

tasks = ['Chr22q', 'MethylationSubgroup', 'Chr1p'] # ['Chr22q', 'MethylationSubgroup', 'Chr1p']

begin_time = time.time()
start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f'\n\nStarted nonlin_loo.py at: {start_time}\n\n')

for task in tasks:
    print(f'\nStarting {task}...')
    exp = NonlinearLOOExp(
        model=model,
        fixed_model_params=fixed_params,
        param_to_sweep=param_to_sweep,
        sweep_values=sweep_values,
        prediction_task=task, 
        output_dir=output_folder,
        use_smote=True,
        feat_file="data/radiomics/features8_smoothed/features_wide.csv", # f"data/collage_sparse/windowsize-9_binsize-64_summary_22nansfilled_pruned.csv" # "data/combined_feats/5-15-24_radiomics_pruned-collage_features.csv"
        feat_select='Mod-AX_3D_T1_POST'
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
