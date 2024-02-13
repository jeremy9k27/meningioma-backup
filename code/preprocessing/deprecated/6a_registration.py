# File: 6a_registration.py
# Date: 02/04/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description: Performs registration on the volumetric MRI scans.

#--------------------------#
####      CONTENTS      ####
#--------------------------#
# N. Notes
# 0. Package imports
# 1. File wrangling
# 2. Registration

#--------------------------#
####      N. NOTES      ####
#--------------------------#
# This script is meant to perform registration on the volumetric MRI scans.
# We will use the ANTsPy package to perform the registration in two steps:
#   1. Nonlinear registration to a AX 3D T1 POST image: aligns all scans of a subject to their own AX 3D T1 POST scan. 
#   2. Rigid registration to MNI template: aligns the scans using only rotation and translation. Uses the MNI ICBM 152 nonlinear atlas version 2009 as the template.
#
# This script relies on the following file(s) as inputs:
#   * data/preprocessing/output/4c_HISTOGRAM_EQUALIZED/*/*_Brainlab/*/*.nii.gz
#
# This script generates the following file(s) as outputs:
#   * data/preprocessing/output/6_REGISTERED/*/*_Brainlab/*/*.nii.gz
#   * data/preprocessing/output/6_REGISTERED/log.txt

#--------------------------#
#### 0. PACKAGE IMPORTS ####
#--------------------------#
from utils import setup, lsdir, read_example_mri
from datetime import datetime
from tqdm import tqdm
import time
import shutil
import os
import ants
import logging

#-------------------------#
#### 1. FILE WRANGLING ####
#-------------------------#
setup()
begin_time = time.time()

data_dir = 'data/preprocessing/output/4c_HISTOGRAM_EQUALIZED'
output_dir = 'data/preprocessing/output/6_REGISTERED'

log_file = os.path.join(output_dir, 'log.txt')

if not os.path.exists(output_dir): os.makedirs(output_dir)

date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
bar = '-' * 80
os.system(f"echo '\n{bar}\n' >> {log_file}")
os.system(f"echo 'Running script 6_registration.py at {date}\n' >> {log_file}")
print(f"Logging output to {log_file}")
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(levelname)s: %(message)s')

nonlinear_template = 'AX_3D_T1_POST'
rigid_template = 'https://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_sym_09a_nifti.zip' # 'MNI ICBM 152 nonlinear atlas version 2009'
rigid_template_zip = rigid_template.split('/')[-1]
rigid_template_dir = rigid_template_zip.split('_nifti')[0]
os.system(f"echo 'Nonlinear template used: {nonlinear_template}' >> {log_file}")
rigid_template_path = f'{output_dir}/{rigid_template_dir}/mni_icbm152_t1_tal_nlin_sym_09a.nii' # is T1w the right one for all my scans?
if not os.path.exists(rigid_template_path):
    os.system(f"cd {output_dir} && wget {rigid_template} && unzip {rigid_template_zip} && rm {rigid_template_zip} && cd ../../../../")
os.system(f"echo 'Rigid template used: {rigid_template_path}\n' >> {log_file}")
mni_template = ants.image_read(rigid_template_path, reorient='IAL')

#-----------------------#
#### 2. REGISTRATION ####
#-----------------------#
for subject in tqdm(lsdir(data_dir), desc="Subjects"):
    for session in tqdm(lsdir(f'{data_dir}/{subject}'), desc="Sessions", leave=False):
        # Check if an AX_3D_T1_POST scan exists for this session
        current_scans = lsdir(f'{data_dir}/{subject}/{session}')
        scan_types = [scan.split('-')[-1] for scan in current_scans]
        ax3dt1post_found = True
        if nonlinear_template not in scan_types: ax3dt1post_found = False
        for scan in tqdm(current_scans, desc="Scans", leave=False):
            cur_input_dir = f'{data_dir}/{subject}/{session}/{scan}'
            cur_output_dir = f'{output_dir}/{subject}/{session}/{scan}'
            scan_type = scan.split('-')[-1]
            if not os.path.exists(f'{cur_output_dir}/{session}_{scan}.nii.gz'): 
                if not os.path.exists(cur_output_dir): os.makedirs(cur_output_dir)
                shutil.copy(f'{cur_input_dir}/{session}_{scan}.json', f'{cur_output_dir}/{session}_{scan}.json')
                original_mri = read_example_mri(data_dir, subject, session, scan, ants=True, orientation='IAL')
                os.system(f"echo 'Processing {session}/{scan}...' >> {log_file}")
                if ax3dt1post_found:
                    # Step 1: Nonlinear registration to AX 3D T1 POST
                    if scan_type != nonlinear_template:
                        os.system(f"echo 'Starting nonlinear registration to {nonlinear_template}...' >> {log_file}")
                        ax3dt1post_path = [s for s in current_scans if s.endswith(nonlinear_template)][0]
                        ax3dt1post = read_example_mri(data_dir, subject, session, ax3dt1post_path, ants=True, orientation='IAL')
                        
                        # Check that ax3dt1post is 1x1x1mm
                        if ax3dt1post.spacing != (1.0, 1.0, 1.0):
                            os.system(f"echo 'Warning: {session}/{ax3dt1post_path} does not have 1x1x1mm spacing, therefore {session}/{scan} will not either. It will have spacing: {ax3dt1post.spacing}' >> {log_file}")
                        
                        type_of_transform = 'SyNRA'
                        try:
                            nonlinear_transform = ants.registration(
                                fixed=ax3dt1post,
                                moving=original_mri,
                                type_of_transform=type_of_transform,
                                verbose=False
                            )
                            step1_output = nonlinear_transform['warpedmovout']
                            step1_output.to_file(f'{cur_output_dir}/{session}_{scan}_{type_of_transform}_registration_to_{nonlinear_template}.nii.gz')
                        except Exception as e:
                            os.system(f"echo 'Error: {e}' >> {log_file}")
                            os.system(f"echo 'Fixed shape: {ax3dt1post.numpy().shape}' >> {log_file}")
                            os.system(f"echo 'Moving shape: {original_mri.numpy().shape}' >> {log_file}")
                            os.system(f"echo 'Unable to register {session}/{scan}...\n' >> {log_file}")
                            continue
                    else:
                        step1_output = original_mri
                else:
                    os.system(f"echo 'Warning: Cannot do nonlinear registration for {session}/{scan} since no AX_3D_T1_POST scan was found for {session}...' >> {log_file}")
                    step1_output = original_mri
                
                # Step 2: Rigid registration to MNI template
                type_of_transform = 'Rigid'
                os.system(f"echo 'Starting {type_of_transform.lower()} registration to {rigid_template_path}...' >> {log_file}")
                try:
                    rigid_transform = ants.registration(
                        fixed=mni_template, # OR ants.image_read(ants.get_ants_data('mni'))
                        moving=step1_output,
                        type_of_transform=type_of_transform,
                        verbose=False
                    )
                    step2_output = rigid_transform['warpedmovout']
                    step2_output.to_file(f'{cur_output_dir}/{session}_{scan}.nii.gz')
                    os.system(f"echo 'Done!\n' >> {log_file}")
                except Exception as e:
                    os.system(f"echo 'Error: {e}' >> {log_file}")
                    os.system(f"echo 'Fixed shape: {mni_template.numpy().shape}' >> {log_file}")
                    os.system(f"echo 'Moving shape: {step1_output.numpy().shape}' >> {log_file}")
                    os.system(f"echo 'Unable to register {session}/{scan}...\n' >> {log_file}")
                    continue
            
time_elapsed = time.time() - begin_time
date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
os.system(f"echo 'Completed 6_registration.py at {date}\n' >> {log_file}")
os.system(f"echo 'Total elapsed time: {time_elapsed}' >> {log_file}")
os.system(f"echo '\n{bar}\n' >> {log_file}")