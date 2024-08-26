# File: 3c_rename_scans.py
# Date: 08/23/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description: This script renames scans based on their thumbnail's location in the 3_RENAMED_SCANS/thumbnails/FINISHED folder

#--------------------------#
####      CONTENTS      ####
#--------------------------#
# N. Notes
# 0. Package imports

#--------------------------#
####      N. NOTES      ####
#--------------------------#
# This script renames scans based on their thumbnail's location in the 3_RENAMED_SCANS/thumbnails/FINISHED folder
#
# This script relies on the following file(s) as inputs:
#   * data/round2_preprocessing/output/2_NIFTI/*
#   * data/round2_preprocessing/SCAN_TYPE_CLEANUP/thumbnails/FINISHED/*
#
# This script generates the following file(s) as outputs:
#   * data/round2_preprocessing/output/3_RENAMED_SCANS/*

#%%------------------------#
#### 0. PACKAGE IMPORTS ####
#--------------------------#
from utils import setup
from tqdm import tqdm
import shutil
import os

setup()

thumb_dir = 'data/round2_preprocessing/SCAN_TYPE_CLEANUP/thumbnails/FINISHED'
origin_dir = 'data/round2_preprocessing/output/2_NIFTI'
dest_dir = 'data/round2_preprocessing/output/3_RENAMED_SCANS'

dirs_of_interest = ['AX_3D_T1_POST', 'AX_ADC', 'AX_DIFFUSION', 'SAG_3D_FLAIR']

def breakdown_filename(filename):
    parts = filename.replace('.png', '').split('-')
    sub_session_num = parts[0]
    scan_name = '-'.join(parts[1:])
    fine_parts = sub_session_num.split('_')
    scan_num = fine_parts[-1]
    subject_num = fine_parts[0]
    session = '_'.join(fine_parts[1:-1])
    return subject_num, session, scan_num, scan_name

for scan_type in tqdm(dirs_of_interest, desc='Scan types', total=len(dirs_of_interest), position=0, smoothing=0.5, dynamic_ncols=True, colour='green'):
    thumbs = [f for f in os.listdir(f'{thumb_dir}/{scan_type}') if f.endswith('.png')]
    for thumb in tqdm(thumbs, desc='Thumbnails', total=len(thumbs), position=1, smoothing=0.5, dynamic_ncols=True, colour='blue'):
        try:
            subject_num, session, scan_num, scan_name = breakdown_filename(thumb)
            
            origin_scan = f'{origin_dir}/{subject_num}/{subject_num}_{session}/{scan_num}-{scan_name}/{subject_num}_{session}_{scan_num}-{scan_name}.nii.gz'
            origin_json = f'{origin_dir}/{subject_num}/{subject_num}_{session}/{scan_num}-{scan_name}/{subject_num}_{session}_{scan_num}-{scan_name}.json'
            cur_dest_dir = f'{dest_dir}/{subject_num}/{subject_num}_{session}/{scan_num}-{scan_type}/'            
            dest_scan = f'{cur_dest_dir}/{subject_num}_{session}_{scan_num}-{scan_type}.nii.gz'
            dest_json = f'{cur_dest_dir}/{subject_num}_{session}_{scan_num}-{scan_type}.json'

            if not os.path.exists(cur_dest_dir): os.makedirs(cur_dest_dir)
            shutil.copy(origin_scan, dest_scan)
            shutil.copy(origin_json, dest_json)

        except Exception as e:
            print(f'Error processing {thumb}: {e}')
            continue