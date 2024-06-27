# %%
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import numpy as np
import pandas as pd
from preprocessing.utils import setup

setup()

def median_abs_deviation(series):
    return np.nanmedian(np.abs(series - np.nanmedian(series)))

df = pd.read_csv('data/4a_radiomics_for_all_seg_sub_pairs/features.csv')

# %%
groups = df.drop(columns=['Subject Providing Scan']).groupby(['Subject Providing Segmentation', 'Segmentation Label', 'Scan Sequence'])

# Calculate the median for each group
medians = groups.median().reset_index().pivot(
        index='Subject Providing Segmentation', 
        columns=['Scan Sequence', 'Segmentation Label']
    )

# Apply the median_abs_deviation function to calculate MAD for each group
mads = groups.apply(lambda g: g.apply(median_abs_deviation)).reset_index().pivot(
        index='Subject Providing Segmentation', 
        columns=['Scan Sequence', 'Segmentation Label']
    )

# %%
medians.columns = [f"Mod-{modality}-SegLab-{segmentation_label}-Feat-{feature}" for (feature, modality, segmentation_label) in medians.columns]
mads.columns = [f"Mod-{modality}-SegLab-{segmentation_label}-Feat-{feature}" for (feature, modality, segmentation_label) in mads.columns]

# %%
output_dir = 'data/4b_processed_seg_sub_pair_feats'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
medians.to_csv(f'{output_dir}/medians.csv')
mads.to_csv(f'{output_dir}/mads.csv')
# %%
