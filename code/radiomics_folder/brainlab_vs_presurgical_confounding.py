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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

setup()

task = 'Chr22q'
data_dir = f'results/LOO_rad10_ultrafine_1-6-25/{task}'
preds_df = pd.read_csv(f'{data_dir}/best_model_preds.csv')
conf_df = pd.read_csv(f'data/labels/subject_sessions.csv')
feats_df = pd.read_csv(f'data/radiomics/features10_smoothed/features_wide.csv')

# %%
preds_by_session = pd.merge(preds_df, conf_df, on='Subject Number')
feats_by_session = pd.merge(feats_df, conf_df, on='Subject Number')
feats_w_preds_by_session = pd.merge(feats_by_session, preds_by_session.drop(columns=['Session']), on='Subject Number', how='left')
nans = np.isnan(feats_w_preds_by_session['Prediction'])
corrects = feats_w_preds_by_session['Label'] == feats_w_preds_by_session['Prediction']
feats_w_preds_by_session.loc[:, 'Correct'] = [np.nan if n else c for n, c in zip(nans, corrects)]

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
# Scale data
feats_scaled = StandardScaler().fit_transform(feats_df.dropna(axis=1, how='all').fillna(0))

#### PCA ####
pca = PCA(n_components=2, random_state=42)
pca_results = pca.fit_transform(feats_scaled)
# pca_results = pca.fit_transform(flattened_embeddings)

var = pca.explained_variance_ratio_

pca_df = pd.DataFrame(pca_results, columns=['PC1', 'PC2'])
# for key in labels.keys():
#     pca_df.loc[:, key] = np.stack(labels[key][:len(pca_results)])

# for key in labels.keys():
pca_df.loc[:, 'Session'] = np.stack(feats_by_session['Session'][:len(pca_results)])
g = sns.relplot(data=pca_df, x='PC1', y='PC2', hue='Session', palette='tab10')
g.figure.suptitle(f'PCA\nVar Exp: PC1={round(var[0], 2)}, PC2={round(var[1], 2)}', y=1.02)
g.savefig(f'data/embeddings/PCA_embeddings.png')

#### UMAP ####
umap_results = umap.UMAP(n_components=2, random_state=42).fit_transform(feats_scaled)

umap_df = pd.DataFrame(umap_results, columns=['x', 'y'])
umap_df.loc[:, 'Session'] = np.stack(feats_by_session['Session'][:len(umap_results)])

# for key in labels.keys():
#     umap_df.loc[:, key] = np.stack(labels[key][:len(umap_results)])

# for key in labels.keys():
g = sns.relplot(data=umap_df, x='x', y='y', hue='Session', palette='tab10')
g.figure.suptitle(f'UMAP', y=1.02)
# g.savefig(f'data/embeddings/UMAP_embeddings.png')

# %%
