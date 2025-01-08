# %%
import pandas as pd
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup, lsdir
from utils import count_subjects, get_subset_scan_counts

setup()

# %%
MRI_DIR = 'data/preprocessing/output/7b_COMPLETED_PREPROCESSED'
subjects = lsdir(MRI_DIR)
sessions = []
for s in subjects:
    sess_name = lsdir(f'{MRI_DIR}/{s}')[0]
    if 'presurgical' in sess_name.lower():
        sessions.append('presurgical')
    elif 'brainlab' in sess_name.lower():
        sessions.append('brainlab')
    else:
        sessions.append('other')

pd.DataFrame({'Subject Number': subjects, 'Session': sessions}).to_csv('data/labels/subject_sessions.csv', index=False)

# %%
df = pd.read_csv('data/radiomics/features6/features_wide.csv')

# %%
_, _, have_df = count_subjects(drop_by_outcome=False)
get_subset_scan_counts(have_df['Subject Number'].to_list())

# %%
print(have_df['MethylationSubgroup'].value_counts())
print(have_df['Chr1p'].value_counts())
print(have_df['Chr22q'].value_counts())
print(have_df['Chr9p'].value_counts())
# %%
