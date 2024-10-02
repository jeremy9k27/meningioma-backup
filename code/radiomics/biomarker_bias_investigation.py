# %% 
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup
import joblib
from LOOExperiment import LOOExperiment
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

setup()

# %%
task = 'Chr22q'
bias_check_task = 'MethylationSubgroup'

preds_df = pd.read_csv(f'results/LOO_rad8_fine_9-6-24/{task}/best_model_preds.csv')
labels_df = pd.read_csv('data/labels/MeningiomaBiomarkerData.csv')
labels_df = labels_df[labels_df['Subject Number'].isin(preds_df['Subject Number'])]
results_df = preds_df.merge(labels_df, on='Subject Number')

print(results_df[task].value_counts())
overall_ba = balanced_accuracy_score(preds_df['Label'].values, preds_df['Prediction'].values)
# overall_auc = roc_auc_score(label_binarize(preds_df['Label'].values, classes=[0, 1]), label_binarize(preds_df['Prediction'].values, classes=[0, 1]), multi_class='ovr', average='weighted')
# %%
print(f'Overall balanced accuracy: {overall_ba.round(3)}')
for lab in [0, 1, 2]:
    sub_group_idx = results_df[results_df[bias_check_task] == lab]['Subject Number'].index
    sub_group_df = results_df.filter(items=sub_group_idx, axis=0)
    print(sub_group_df[task].value_counts())
    sub_group_ba = balanced_accuracy_score(sub_group_df['Label'].values, sub_group_df['Prediction'].values)
    # sub_group_auc = roc_auc_score(label_binarize(sub_group_df['Label'].values, classes=[0, 1, 2]), label_binarize(sub_group_df['Prediction'].values, classes=[0, 1, 2]), multi_class='ovr', average='weighted')
    print(f'Balanced accuracy for subgroup {lab}: {sub_group_ba.round(3)}')

# %%
print(results_df['MethylationSubgroup'].corr(results_df['Chr22q']))
print(results_df['MethylationSubgroup'].corr(results_df['Chr1p']))
print(results_df['Chr22q'].corr(results_df['Chr1p']))
# %%
