# %%
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import setup, lsdir, read_example_mri
from datetime import datetime
import time
import shutil
import os
import ants
import logging
from tqdm import tqdm
import numpy as np

setup()

data_dir = 'data/preprocessing/output/6c_NONLIN_WARP_REGISTERED'

mni_template_path = 'data/preprocessing/output/6c_NONLIN_WARP_REGISTERED/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii'
mni_template = ants.image_read(mni_template_path, reorient='IAL')

# %%
def fix_file_extension(filename):
    if os.path.exists(filename):
        new_filename = filename[:-3] + '.nii.gz'
        os.rename(filename, new_filename)
        print(f'Renamed: {filename} to {new_filename}')
    else:
        print(f'No change needed for: {filename}')

def get_segs_for_subject(sub_no, segs_dirs=['data/smoothed_segmentations/', 'data/segmentations/']):
    segs = []
    for segs_dir in segs_dirs:
        s = [segs_dir + f for f in os.listdir(segs_dir) if f' {sub_no} ' in f or f' {sub_no}.' in f]
        segs.extend(s)
    return segs

def prop_tx_to_seg(seg, tx, output_path):
    prop_seg = ants.apply_transforms(
        fixed=mni_template,
        moving=seg,
        transformlist=[tx],
        verbose=False
    )
    prop_seg.to_file(f'{output_path}')

# %%
subjects = [s for s in lsdir(data_dir) if s.isdigit()]
# for subject in subjects:
#     brainlab = lsdir(f'{data_dir}/{subject}/')[0]
#     scans = lsdir(f'{data_dir}/{subject}/{brainlab}/')
#     t1post_idx = np.where(['AX_3D_T1_POST' in s for s in scans])[0][0]
#     t1post = scans[t1post_idx]
#     tx_file = f'{brainlab}_{t1post}_SyNRA_to_MNI_transform_tx_0.gz'
#     tx_path = f'{data_dir}/{subject}/{brainlab}/{t1post}/{tx_file}'
#     fix_file_extension(tx_path)

# %%
for subject in tqdm(subjects, desc='Subjects', total=len(subjects), smoothing=0, dynamic_ncols=True):
    brainlab = lsdir(f'{data_dir}/{subject}/')[0]
    scans = lsdir(f'{data_dir}/{subject}/{brainlab}/')
    t1post_idx = np.where(['AX_3D_T1_POST' in s for s in scans])[0][0]
    t1post = scans[t1post_idx]
    tx_file = f'{brainlab}_{t1post}_SyNRA_to_MNI_transform_tx_0.nii.gz'
    tx_path = f'{data_dir}/{subject}/{brainlab}/{t1post}/{tx_file}'
    segs_list = get_segs_for_subject(subject)
    for seg in tqdm(segs_list, desc='Segs', total=len(segs_list), smoothing=0, dynamic_ncols=True, leave=False, position=1):
        seg_name = seg.split('/')[-1]
        if 'Smoothed' in seg_name:
            output_dir = f'data/8_mni_registered_smoothed_segs'
        else:
            output_dir = f'data/8_mni_registered_segs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = f'{output_dir}/{seg_name}'
        prop_tx_to_seg(ants.image_read(seg, reorient='IAL'), tx_path, output_path)
# %%
