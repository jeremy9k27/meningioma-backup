# %%
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup
from utils import read_ndarray, write_ndarray
import numpy as np
import joblib
from tqdm import tqdm

setup()

# sparse_data_dir = 'data/collage_sparse/windowsize-9_binsize-64'
old_data_dir = 'data/old_collage_large/windowsize-9_binsize-64'
test_dir = 'data/test/'
old_files = [f for f in os.listdir(old_data_dir) if f.endswith('.joblib')]

def check_data_integrity(old_arr, new_arr):
    old_val_idxs = np.stack(np.where(~np.isnan(old_arr)))
    new_val_idxs = np.stack(np.where(~np.isnan(new_arr)))
    return {
        'allclose': np.allclose(old_arr, new_arr, equal_nan=True),
        'shape_match': old_arr.shape == new_arr.shape,
        'dtype_match': old_arr.dtype == new_arr.dtype,
        'num_nans_old': np.isnan(old_arr).sum(),
        'num_nans_new': np.isnan(new_arr).sum(),
        'num_nans_diff': np.isnan(new_arr).sum() - np.isnan(old_arr).sum(),
        'num_zeros_old': np.sum(old_arr == 0),
        'num_nans_match': np.isnan(old_arr).sum() == np.isnan(new_arr).sum(),
        'locations_of_vals_match': np.array_equal(old_val_idxs, new_val_idxs)
    }

# %%
for fname in tqdm(old_files, total=len(old_files)):
    old_arr = joblib.load(f'{old_data_dir}/{fname}')
    write_ndarray(old_arr, f'{test_dir}/{fname}')
# %%
