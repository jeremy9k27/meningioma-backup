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

medians = pd.read_csv('data/4b_processed_seg_sub_pair_feats/medians.csv')
mads = pd.read_csv('data/4b_processed_seg_sub_pair_feats/mads.csv')
radiomics = pd.read_csv('data/radiomics/features8_smoothed/features_wide.csv')
# %%
medians.rename(columns={'Subject Providing Segmentation': 'Subject Number'}, inplace=True)
mads.rename(columns={'Subject Providing Segmentation': 'Subject Number'}, inplace=True)
# %%
medians = medians[radiomics.columns]
mads = mads[radiomics.columns]

# %%
rad_subs = radiomics['Subject Number']
med_subs = medians['Subject Number']
mad_subs = mads['Subject Number']
assert (rad_subs == med_subs).all() and (rad_subs == mad_subs).all()

# %%
radiomics = radiomics.drop(columns='Subject Number')
medians = medians.drop(columns='Subject Number')
mads = mads.drop(columns='Subject Number')

# Step 1: Create a mask where mads is not 0 or NaN
mask = (mads != 0) & (~mads.isna())

# Step 2: Perform the calculation (radiomics - medians) / mads where mask is True
result = radiomics.copy()  # Initialize result DataFrame
result[mask] = (radiomics[mask] - medians[mask]) / mads[mask]

# %%
output_dir = 'data/4c_radiomics_adjusted_w_medians_and_mads'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

result['Subject Number'] = rad_subs
result = result[['Subject Number'] + result.columns[:-1].tolist()]
result.to_csv(os.path.join(output_dir, 'features_wide.csv'), index=False)