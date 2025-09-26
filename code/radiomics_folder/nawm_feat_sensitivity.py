# %% 
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup
from utils import clean_feature_names
import pandas as pd
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
