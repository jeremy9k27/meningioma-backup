# %%
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup
import pandas as pd
import numpy as np
import re

setup()

radiomics_df = pd.read_csv('data/radiomics/features6/features_wide.csv')
collage_df = pd.read_csv('data/collage_sparse/windowsize-9_binsize-64_summary_22nansfilled.csv')

important_feats = [f for f in collage_df.columns if 'skewness' in f or 'kurtosis' in f or 'entropy' in f]

pruned_collage_df = collage_df[['Subject Number'] + important_feats]
pruned_collage_df.to_csv('data/collage_sparse/windowsize-9_binsize-64_summary_22nansfilled_pruned.csv', index=False)
# %%

print('Radiomics shape: ', radiomics_df.shape)
print('Pruned collage shape: ', pruned_collage_df.shape)

combined_df = pd.merge(radiomics_df, pruned_collage_df, on=['Subject Number'], how='outer')

print('Combined shape: ', combined_df.shape)

if not os.path.exists('data/combined_feats'): os.makedirs('data/combined_feats')

combined_df.to_csv('data/combined_feats/5-15-24_radiomics_pruned-collage_features.csv', index=False)
# %%
