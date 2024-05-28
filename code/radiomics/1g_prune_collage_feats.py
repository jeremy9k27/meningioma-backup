# %%
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup
import pandas as pd

setup()

dd = 'data/collage_sparse_small_windows'
win_size = 3
bin_size = 32

radiomics_df = pd.read_csv('data/radiomics/features8_smoothed/features_wide.csv')
collage_df = pd.read_csv(f'{dd}/windowsize-{win_size}_binsize-{bin_size}_summary_22nansfilled.csv')

important_feats = [f for f in collage_df.columns if 'skewness' in f or 'kurtosis' in f or 'entropy' in f]

pruned_collage_df = collage_df[['Subject Number'] + important_feats]
pruned_collage_df.to_csv(f'{dd}/windowsize-{win_size}_binsize-{bin_size}_summary_22nansfilled_pruned.csv', index=False)
# %%

print('Radiomics shape: ', radiomics_df.shape)
print('Pruned collage shape: ', pruned_collage_df.shape)

combined_df = pd.merge(radiomics_df, pruned_collage_df, on=['Subject Number'], how='outer')

print('Combined shape: ', combined_df.shape)

if not os.path.exists('data/combined_feats'): os.makedirs('data/combined_feats')

combined_df.to_csv(f'data/combined_feats/radiomics8-smoothed_pruned-collage-ws-{win_size}-bs-{bin_size}_features.csv', index=False)
# %%
