# %%
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef, jaccard_score

setup()

task = 'Chr22q'
data_dir = f'results/LOO_rad10_ultrafine_1-6-25/{task}'
preds_df = pd.read_csv(f'{data_dir}/best_model_preds.csv')
conf_df = pd.read_csv(f'data/labels/subject_sessions.csv')

# %%
preds_by_session = pd.merge(preds_df, conf_df, on='Subject Number')

# %%
def get_binary_metrics(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    metrics = {
        'Binary F1': f1_score(y_true, y_pred, average='binary'),
        'Weighted F1': f1_score(y_true, y_pred, average='weighted'),
        'Binary Precision': precision_score(y_true, y_pred, average='binary'),
        'Weighted Precision': precision_score(y_true, y_pred, average='weighted'),
        'Binary Recall (Sensitivity)': recall_score(y_true, y_pred, average='binary'),
        'Weighted Recall (Sensitivity)': recall_score(y_true, y_pred, average='weighted'),
        'Specificity': tn / (tn + fp),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Binary Jaccard': jaccard_score(y_true, y_pred, average='binary'),
        'Weighted Jaccard': jaccard_score(y_true, y_pred, average='weighted')
    }
    return metrics

def get_multiclass_metrics(y_true, y_pred):
    metrics = {
        'Macro F1': f1_score(y_true, y_pred, average='macro'),
        'Micro F1': f1_score(y_true, y_pred, average='micro'),
        'Weighted F1': f1_score(y_true, y_pred, average='weighted'),
        'Macro Precision': precision_score(y_true, y_pred, average='macro'),
        'Micro Precision': precision_score(y_true, y_pred, average='micro'),
        'Weighted Precision': precision_score(y_true, y_pred, average='weighted'),
        'Macro Recall (Sensitivity)': recall_score(y_true, y_pred, average='macro'),
        'Micro Recall (Sensitivity)': recall_score(y_true, y_pred, average='micro'),
        'Weighted Recall (Sensitivity)': recall_score(y_true, y_pred, average='weighted'),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Macro Jaccard': jaccard_score(y_true, y_pred, average='macro'),
        'Micro Jaccard': jaccard_score(y_true, y_pred, average='micro'),
        'Weighted Jaccard': jaccard_score(y_true, y_pred, average='weighted')
    }

    return metrics

# %% 
brainlab_preds = preds_by_session[preds_by_session['Session'] == 'brainlab']
presurgical_preds = preds_by_session[preds_by_session['Session'] == 'presurgical']

if task == 'MethylationSubgroup':
    brainlab_metrics = get_multiclass_metrics(brainlab_preds['Label'].values, brainlab_preds['Prediction'].values)
    presurgical_metrics = get_multiclass_metrics(presurgical_preds['Label'].values, presurgical_preds['Prediction'].values)
else:
    brainlab_metrics = get_binary_metrics(brainlab_preds['Label'].values, brainlab_preds['Prediction'].values)
    presurgical_metrics = get_binary_metrics(presurgical_preds['Label'].values, presurgical_preds['Prediction'].values)

# %%
