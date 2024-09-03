# File: 3d_count_scan_types.py
# Date: 08/28/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description: Counts the number of each scan type for all subjects/sessions in a given data dir.

#--------------------------#
####      CONTENTS      ####
#--------------------------#
# N. Notes
# 0. Package imports

#--------------------------#
####      N. NOTES      ####
#--------------------------#
# This script counts the number of each scan type for all subjects/sessions in a given data dir.
#
# This script relies on the following file(s) as inputs:
#   * data/round2_preprocessing/output/3_RENAMED_SCANS

#%%------------------------#
#### 0. PACKAGE IMPORTS ####
#--------------------------#
from utils import setup, lsdir
from collections import Counter
setup()

data_dir = 'data/round2_preprocessing/output/3_RENAMED_SCANS' # 'data/round2_preprocessing/output/3_RENAMED_SCANS' # 'data/preprocessing/output/2_NIFTI'
subjects = lsdir(data_dir)
all_scan_types = []
for s in subjects:
    sessions = lsdir(f'{data_dir}/{s}')
    for sess in sessions:
        scans = lsdir(f'{data_dir}/{s}/{sess}')
        scan_types = [scan.split('-')[-1] for scan in scans]
        # assert ('AX_DIFFUSION' in scan_types) == ('AX_ADC' in scan_types), f'{sess}: {scan_types}'
        all_scan_types.extend(scan_types)
        print(f'{sess}: {Counter(scan_types)}')

print(f'All scan counts for n = {len(subjects)} subjects: {Counter(all_scan_types)}')
# %%
