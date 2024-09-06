# %% 
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup
import joblib
from LOOExperiment import LOOExperiment
from sklearn.preprocessing import StandardScaler
from utils import clean_feature_names
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

setup()

# read in radiomics data
radiomics_df = pd.read_csv('data/radiomics/features9_smoothed/features_wide.csv')

# extract NAWM features and the subject number
nawm_df = radiomics_df[['Subject Number'] + [col for col in radiomics_df.columns if '-7-' in col]]

# Drop all meaningless shape features, all features that are all NaNs or only 1 unique value, and all rows that are all NaNs
nawm_df = nawm_df.drop(
    columns=[col for col in nawm_df.columns if 'shape' in col]
).dropna(
    axis=1, how='all'
).drop(
    columns=[col for col in nawm_df.columns if nawm_df[col].nunique() == 1]
).dropna()

# clean up column names
nawm_df.columns = clean_feature_names(nawm_df.columns)

# %%
rob_feats_df = pd.read_csv('data/5a2_RoB_feats_using_9-4-24_smooth_segs/features.csv')
rob_feats_df['Segmentation Label'] = 7
rob_feats_wide = rob_feats_df.pivot(
    index='Subject Number', 
    columns=['Scan Sequence', 'Segmentation Label']
)
rob_feats_wide.columns = clean_feature_names([f"Mod-{modality}-SegLab-{segmentation_label}-Feat-{feature}" for (feature, modality, segmentation_label) in rob_feats_wide.columns])
rob_feats_wide['Subject Number'] = rob_feats_wide.index

# %%
rob_final = rob_feats_wide[nawm_df.columns]
rob_final = rob_final[rob_final['Subject Number'].isin(nawm_df['Subject Number'])].reset_index(drop=True)

nawm_final = nawm_df.reset_index(drop=True)

# %%
rel_err = abs(nawm_final - rob_final) / abs(nawm_final)
rel_err = rel_err.drop(columns=['Subject Number'])
rel_err = rel_err.reindex(rel_err.median().sort_values(ascending=False).index, axis=1)

# %% Box and whisker plots
def make_boxplot(cols=range(10)):
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(10, 10))
    sns.boxplot(data=rel_err.iloc[:, cols], orient='h', showfliers=False)
    plt.title('Feature Sensitivity: abs(NAWM - RoB)/abs(NAWM)')
    plt.show()

make_boxplot(range(10))
make_boxplot(range(170, 184))
# make_boxplot(range(10, 20))
# make_boxplot(range(20, 30))
# make_boxplot(range(670, 680))

# %% Line graph
median_df = pd.DataFrame(rel_err.median().sort_values(ascending=False), columns=['Median']).reset_index()
median_df = median_df.rename(columns={'index': 'Feature'})

sns.set_theme(style='whitegrid')
plt.figure(figsize=(24, 8))
ax = sns.lineplot(data=median_df, x='Feature', y='Median', marker='o')
plt.title('Median Difference in Feature Values')
plt.xlabel('Feature')
plt.ylabel('Median Difference Values abs(NAWM - RoB)/abs(NAWM)')
plt.xticks(rotation=45)
# turn off x axis labels
ax.set_xticklabels([])
ax.set_yscale('log')
plt.show()

# %%
task = 'MethylationSubgroup'

def prep_data_for_loocv(features_file='data/radiomics/features9_smoothed/features_wide.csv', labels_file='data/labels/MeningiomaBiomarkerData.csv', outcome=task, scaler_obj=StandardScaler(), feat_select=None):
    # read in features and labels, merge
    features = pd.read_csv(features_file)
    labels = pd.read_csv(labels_file)
    labels = labels.dropna(subset=[outcome])
    labels = labels[labels['Subject Number'].isin(features['Subject Number'])]
    data = features.merge(labels, on='Subject Number')
    data.columns = clean_feature_names(data.columns)
    data = data.dropna(axis=1, how='all').fillna(0)
    subject_numbers = data['Subject Number']
    X = data.drop(columns=['Subject Number', 'MethylationSubgroup', 'Chr1p', 'Chr22q', 'Chr9p', 'TERT'])
    y = data[outcome].values.astype(int)

    # scale data if specified
    if scaler_obj is not None:
        X = pd.DataFrame(scaler_obj.fit_transform(X), columns=X.columns)
    
    if feat_select is not None:
        feats_of_interest = [col for col in X.columns if col.startswith(feat_select)]
        X = X[feats_of_interest]
    
    # drop NAWM features
    X = X.drop(columns=[col for col in X.columns if '-7-' in col])
    # X['subject_ID'] = X.apply(create_hash, axis=1)
    return X, y, subject_numbers

# %%
exp = joblib.load(f'results/LOO_smoothest_radiomics9_fine_noNAWM_9-4-24/{task}/exp.pkl')
y_star = exp.y_test
y_hat = np.argmax(exp.test_probs, axis=1)
_, _, subject_numbers = prep_data_for_loocv()

results_df = pd.DataFrame({'Subject Number': subject_numbers, 'y_star': y_star, 'y_hat': y_hat})

overall_ba = balanced_accuracy_score(y_star, y_hat)

labels = pd.read_csv('data/labels/MeningiomaBiomarkerData.csv')
labels = labels[labels['Subject Number'].isin(subject_numbers)]

results_df = results_df.merge(labels, on='Subject Number')
# %%
chr22q_lost_subs = labels[labels['Chr22q'] == 1]['Subject Number'].index
chr22q_intact_subs = labels[labels['Chr22q'] == 0]['Subject Number'].index
chr1p_lost_subs = labels[labels['Chr1p'] == 1]['Subject Number'].index
chr1p_intact_subs = labels[labels['Chr1p'] == 0]['Subject Number'].index
# %%
results_df.filter(items=chr22q_lost_subs, axis=0)
# %%
