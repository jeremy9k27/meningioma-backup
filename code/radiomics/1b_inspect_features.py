# %%
import pandas as pd
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup
from utils import count_subjects, get_subset_scan_counts

setup()

df = pd.read_csv('data/radiomics/features3/features_wide.csv')

# %%
_, _, have_df = count_subjects(drop_by_outcome=False)
get_subset_scan_counts(have_df['Subject Number'].to_list())
# %%
