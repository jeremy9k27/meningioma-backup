# %%
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup
from utils import read_ndarray
import numpy as np

setup()

data_dir = 'data/collage_sparse/windowsize-9_binsize-64'
new_data_dir = 'data/collage_sparse_small_windows/windowsize-3_binsize-32'

file = 'subject-117_SAG_3D_FLAIR_seg-22.joblib'
old = f'{data_dir}/{file}'
new = f'{new_data_dir}/{file}'

old_data = read_ndarray(old)
new_data = read_ndarray(new)

old_data_nonans = np.nan_to_num(old_data)
new_data_nonans = np.nan_to_num(new_data)
print(np.linalg.norm(old_data_nonans - new_data_nonans) / np.linalg.norm(new_data_nonans))
print(np.linalg.norm(old_data_nonans - new_data_nonans) / np.linalg.norm(old_data_nonans))

# %%
