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

dd = 'data/collage_sparse_small_windows'
win_size = 5
bin_size = 64
df = pd.read_csv(f'{dd}/windowsize-{win_size}_binsize-{bin_size}_summary.csv')

# %%
cols_to_fix = [c for c in df.columns if '-22-' in c]
df2 = df.copy()
num_instances_w_more_than_one_present = 0

for c in cols_to_fix:
    c_pieces = c.split('-22-')
    pattern = c_pieces[0] + '-\d+-' + c_pieces[1]
    matches = [s for s in df2.columns if re.match(pattern, s) and '-22-' not in s]
    for i in range(len(df2)):
        if np.isnan(df2[c].iloc[i]):
            present_cols = np.where(~np.isnan(df2[matches].iloc[i]))
            if present_cols[0].size > 0:
                if present_cols[0].size > 1: num_instances_w_more_than_one_present += 1
                print(f"c: {c}, fix: {matches[present_cols[0][0]]}")
                df2.loc[i, c] = df2.loc[i, matches[present_cols[0][0]]]

print('Number of instances with more than one present:', num_instances_w_more_than_one_present)

# %%
print("Before vs. after percent present stats:")
for i in range(len(df)):
    print(f'Row {i}: Before = {round(np.sum(~np.isnan(df.iloc[i]))/df.shape[1], 2)}, After = {round(np.sum(~np.isnan(df2.iloc[i]))/df2.shape[1], 2)}')

print("Percent missingness BEFORE fix for cols to fix: ", np.sum(np.sum(np.isnan(df[cols_to_fix]))) / (df.shape[0] * len(cols_to_fix)))
print("Percent missingness AFTER fix for cols to fix: ", np.sum(np.sum(np.isnan(df2[cols_to_fix]))) / (df2.shape[0] * len(cols_to_fix)))

# %%
df2.to_csv(f'{dd}/windowsize-{win_size}_binsize-{bin_size}_summary_22nansfilled.csv', index=False)

# %%
radiomics_df = pd.read_csv('data/radiomics/features8_smoothed/features_wide.csv')
collage_df = pd.read_csv(f'{dd}/windowsize-{win_size}_binsize-{bin_size}_summary_22nansfilled.csv')

print('Radiomics shape: ', radiomics_df.shape)
print('Collage shape: ', collage_df.shape)

combined_df = pd.merge(radiomics_df, collage_df, on=['Subject Number'], how='outer')

print('Combined shape: ', combined_df.shape)

if not os.path.exists('data/combined_feats'): os.makedirs('data/combined_feats')

combined_df.to_csv(f'data/combined_feats/radiomics8-smoothed_collage-ws-{win_size}-bs-{bin_size}_features.csv', index=False)
# %%
