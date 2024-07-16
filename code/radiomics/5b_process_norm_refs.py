# %%
# Package imports
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup
import pandas as pd
import numpy as np

setup()

# %%
# Read in normalization reference data
df = pd.read_csv('data/5a_radiomic_normalization_references_constrainedByBrainMask/features.csv')
# Drop columns with only one unique value (no need to normalize using these references)
df_sm = df.drop(columns=df.columns[np.where(df.nunique() == 1)])
# Read in the original radiomics features
pyrad_df = pd.read_csv('data/radiomics/features8_smoothed/features.csv')

# Get a list of variables that are in the pyradiomics data but not in the normalization reference data 
# (E.g., shape features which weren't extracted for this, or the columns we just dropped due to no variation.
# These will be added back as NaNs)
missing_vars = list(set(pyrad_df.columns) - set(df_sm.columns) - set(['Modality', 'Segmentation Label']))
# Add these variables back to the normalization reference data as NaNs
df_sm[missing_vars] = np.nan

# Get a list of segmentation labels from the pyradiomics data
seg_labs = sorted(pyrad_df['Segmentation Label'].unique())
# Expand the normalization reference data to include all combinations of subject number, scan sequence, and segmentation label
seg_labs_df = pd.DataFrame(seg_labs, columns=['Segmentation Label'])
expanded_df = df_sm.merge(seg_labs_df, how='cross')

# Reorder the columns so metadata columns are first
metadata_cols = ['Subject Number', 'Scan Sequence', 'Segmentation Label']
rest_of_cols = [col for col in expanded_df.columns if col not in ['Subject Number', 'Scan Sequence', 'Segmentation Label']]
ref_df = expanded_df[metadata_cols + rest_of_cols]

# %%
# Rename 'Modality' in pyrad_df to 'Scan Sequence' to match the normalization reference data
pyrad_df = pyrad_df.rename(columns={'Modality': 'Scan Sequence'})
# Reorder the rows and columns in pyrad_df to match the normalization reference data
pyrad_df = pyrad_df.sort_values(by=['Subject Number', 'Scan Sequence', 'Segmentation Label'])
pyrad_df = pyrad_df[metadata_cols + rest_of_cols]

ref_df = ref_df.sort_values(by=['Subject Number', 'Scan Sequence', 'Segmentation Label'])
ref_df = ref_df[metadata_cols + rest_of_cols]

# Pivot both dataframes into wide format
pyrad_df_wide = pyrad_df.pivot(index='Subject Number', columns=['Scan Sequence', 'Segmentation Label'])
pyrad_df_wide.columns = [f"Mod-{modality}-SegLab-{segmentation_label}-Feat-{feature}" for (feature, modality, segmentation_label) in pyrad_df_wide.columns]

ref_df_wide = ref_df.pivot(index='Subject Number', columns=['Scan Sequence', 'Segmentation Label'])
ref_df_wide.columns = [f"Mod-{modality}-SegLab-{segmentation_label}-Feat-{feature}" for (feature, modality, segmentation_label) in ref_df_wide.columns]

# Replace any 0s in the reference data with NaNs
ref_df_wide = ref_df_wide.replace(0, np.nan)

# %% Divide the pyradiomics data by the reference data, ignoring NaNs
divided_df = pyrad_df_wide.div(ref_df_wide)

# %% Save the divided data to a CSV file
if not os.path.exists('data/5b_processed_normalized_features'): os.makedirs('data/5b_processed_normalized_features')
divided_df.to_csv('data/5b_processed_normalized_features/features8_smoothed_constrainedByBrainMask_wide.csv')
# %%
