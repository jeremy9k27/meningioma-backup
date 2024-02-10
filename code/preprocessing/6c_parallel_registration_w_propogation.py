# File: 6c_parallel_registration_w_propogation.py
# Date: 02/10/2024
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
#   * data/preprocessing/output/5a_SKULLSTRIPPED/*/*_Brainlab/*/*.nii.gz
#
# This script generates the following file(s) as outputs:
#   * data/preprocessing/output/6c_REGISTERED/*/*_Brainlab/*/*.nii.gz
#   * data/preprocessing/output/6c_REGISTERED/log.txt

#--------------------------#
#### 0. PACKAGE IMPORTS ####
#--------------------------#
from concurrent.futures import ProcessPoolExecutor
from utils import setup, lsdir, read_example_mri
from datetime import datetime
import time
import shutil
import os
import ants
import logging

#-------------------------#
#### 1. FILE WRANGLING ####
#-------------------------#
setup()

data_dir = 'data/preprocessing/output/4c_HISTOGRAM_EQUALIZED'
output_dir = 'data/preprocessing/output/6c_REGISTERED'
skullstrip_dir = 'data/preprocessing/output/5a_SKULLSTRIPPED' # set this to None if you don't want to use skullstripped intermediary images for SWI and DWI scans
log_dir = f'{output_dir}/logfiles'

if not os.path.exists(output_dir): os.makedirs(output_dir)
if not os.path.exists(log_dir): os.makedirs(log_dir)

nonlinear_template = 'AX_3D_T1_POST'
rigid_template = 'https://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_sym_09a_nifti.zip' # 'MNI ICBM 152 nonlinear atlas version 2009'
rigid_template_zip = rigid_template.split('/')[-1]
rigid_template_dir = rigid_template_zip.split('_nifti')[0]

rigid_template_path = f'{output_dir}/{rigid_template_dir}/mni_icbm152_t1_tal_nlin_sym_09a.nii'
if not os.path.exists(rigid_template_path):
    os.system(f"cd {output_dir} && wget {rigid_template} && unzip {rigid_template_zip} && rm {rigid_template_zip} && cd ../../../../")

mni_template = ants.image_read(rigid_template_path, reorient='IAL')

overall_log_file = os.path.join(output_dir, 'log.txt')
print(f"Logging output to {overall_log_file}")
overall_begin_time = time.time()
overall_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
bar = '-' * 80
os.system(f"echo '\n{bar}\n' >> {overall_log_file}")
os.system(f"echo 'Running script 6c_parallel_registration_w_propogation.py at {overall_start_time}\n' >> {overall_log_file}")
os.system(f"echo 'Skull stripping directory used: {skullstrip_dir}' >> {overall_log_file}")
os.system(f"echo 'Nonlinear template used: {nonlinear_template}' >> {overall_log_file}")
os.system(f"echo 'Rigid template used: {rigid_template_path}\n' >> {overall_log_file}")

#-----------------------#
#### 2. REGISTRATION ####
#-----------------------#
def register_subject(subject):
    log_file = os.path.join(log_dir, f'{subject}-log.txt')
    begin_time = time.time()

    for session in lsdir(f'{data_dir}/{subject}'):

        current_scans = lsdir(f'{data_dir}/{subject}/{session}')
        scan_types = [scan.split('-')[-1] for scan in current_scans]

        has_post = False
        has_swi = False
        has_dwi = False
        has_pre = False
        has_flair = False
        has_t2 = False
        if 'AX_3D_T1_POST' in scan_types: has_post = True
        if 'AX_SWI' in scan_types: has_swi = True
        if 'AX_DIFFUSION' in scan_types: has_dwi = True
        if 'AX_3D_T1_PRE' in scan_types: has_pre = True
        if 'SAG_3D_FLAIR' in scan_types: has_flair = True
        if 'SAG_3D_T2' in scan_types: has_t2 = True

        if not has_post:
            os.system(f"echo 'Warning: No AX_3D_T1_POST scan found for {session}, therefore skipping session: {session}' >> {log_file}")
            continue
        else:
            os.system(f"echo 'Starting registration for the session: {session}' >> {log_file}")

        pre_req_scans = current_scans
        pre_req_scans = [s for s in pre_req_scans if not s.endswith('AX_3D_T1_POST')]
        if has_swi: pre_req_scans = [s for s in pre_req_scans if not s.endswith('AX_SWI')]
        if has_dwi: pre_req_scans = [s for s in pre_req_scans if not s.endswith('AX_DIFFUSION')]

        swi_intermediary = None
        if has_swi and has_flair: 
            swi_intermediary = 'SAG_3D_FLAIR'
        elif has_swi and has_pre: 
            swi_intermediary = 'AX_3D_T1_PRE'
        elif has_swi and has_t2:
            swi_intermediary = 'SAG_3D_T2'
        
        dwi_intermediary = None
        if has_dwi and has_flair:
            dwi_intermediary = 'SAG_3D_FLAIR'
        elif has_dwi and has_pre:
            dwi_intermediary = 'AX_3D_T1_PRE'
        elif has_dwi and has_t2:
            dwi_intermediary = 'SAG_3D_T2'

        ax3dt1post_path = [s for s in current_scans if s.endswith(nonlinear_template)][0]
        ax3dt1post = read_example_mri(data_dir, subject, session, ax3dt1post_path, ants=True, orientation='IAL')
        if ax3dt1post.spacing != (1.0, 1.0, 1.0):
            os.system(f"echo '\tWarning: {session}/{ax3dt1post_path} does not have 1x1x1mm spacing. Instead it has: {ax3dt1post.spacing}' >> {log_file}")
        
        swi_intermediary_transform = None
        dwi_intermediary_transform = None

        type_of_transform = 'Rigid'
        os.system(f"echo '\tPerforming {type_of_transform.lower()} registration to {rigid_template_path.split('/')[-1]} for scan {nonlinear_template}' >> {log_file}")
        rigid_transform = ants.registration(
            fixed=mni_template,
            moving=ax3dt1post,
            type_of_transform=type_of_transform,
            verbose=False
        )

        cur_output_dir = f'{output_dir}/{subject}/{session}/{ax3dt1post_path}'
        if not os.path.exists(cur_output_dir): os.makedirs(cur_output_dir)
        rigid_transform['warpedmovout'].to_file(f'{cur_output_dir}/{session}_{ax3dt1post_path}.nii.gz')
        shutil.copy(f'{data_dir}/{subject}/{session}/{ax3dt1post_path}/{session}_{ax3dt1post_path}.json', f'{cur_output_dir}/{session}_{ax3dt1post_path}.json')

        os.system(f"echo '\tNonlinear warping the pre-requisite scans {pre_req_scans} to {ax3dt1post_path}, then propogating rigid registration ({ax3dt1post_path} -> {rigid_template_path.split('/')[-1]}) onto each scan' >> {log_file}")
        for scan in pre_req_scans:
            cur_input_dir = f'{data_dir}/{subject}/{session}/{scan}'
            cur_output_dir = f'{output_dir}/{subject}/{session}/{scan}'
            scan_type = scan.split('-')[-1]
            if not os.path.exists(f'{cur_output_dir}/{session}_{scan}.nii.gz'): 
                os.system(f"echo '\t\tProcessing: {scan}' >> {log_file}")
                if not os.path.exists(cur_output_dir): os.makedirs(cur_output_dir)
                shutil.copy(f'{cur_input_dir}/{session}_{scan}.json', f'{cur_output_dir}/{session}_{scan}.json')
                original_mri = read_example_mri(data_dir, subject, session, scan, ants=True, orientation='IAL')

                try:
                    os.system(f"echo '\t\t\tNonlinear warping {scan} to {ax3dt1post_path}' >> {log_file}")
                    type_of_transform = 'SyNRA'
                    nonlinear_transform = ants.registration(
                        fixed=ax3dt1post,
                        moving=original_mri,
                        type_of_transform=type_of_transform,
                        verbose=False
                    )
                    nonlinear_transform['warpedmovout'].to_file(f'{cur_output_dir}/{session}_{scan}_{type_of_transform}_registration_to_{nonlinear_template}.nii.gz')
                    if scan_type == swi_intermediary: swi_intermediary_transform = nonlinear_transform['fwdtransforms']
                    if scan_type == dwi_intermediary: dwi_intermediary_transform = nonlinear_transform['fwdtransforms']

                    os.system(f"echo '\t\t\tPropogating the rigid registration from {ax3dt1post_path} -> {rigid_template_path.split('/')[-1]} onto {scan}' >> {log_file}")
                    propogated_rigid_transform = ants.apply_transforms(
                        fixed=mni_template,
                        moving=nonlinear_transform['warpedmovout'],
                        transformlist=rigid_transform['fwdtransforms'],
                        verbose=False
                    )

                    propogated_rigid_transform.to_file(f'{cur_output_dir}/{session}_{scan}.nii.gz')

                except Exception as e:
                    os.system(f"echo '\t\t\tError: {e}' >> {log_file}")
                    os.system(f"echo '\t\t\tFixed shape: {ax3dt1post.numpy().shape}' >> {log_file}")
                    os.system(f"echo '\t\t\tMoving shape: {original_mri.numpy().shape}' >> {log_file}")
                    os.system(f"echo '\t\t\tUnable to register {scan}\n' >> {log_file}")
        
        swi_dwi_scans = []
        if has_swi:
            swi_path = [s for s in current_scans if s.endswith('AX_SWI')][0]
            swi_dwi_scans.append(swi_path)
        if has_dwi:
            dwi_path = [s for s in current_scans if s.endswith('AX_DIFFUSION')][0]
            swi_dwi_scans.append(dwi_path)
        
        if len(swi_dwi_scans) > 0:
            os.system(f"echo '\tNonlinear warping {swi_dwi_scans} scans to {ax3dt1post_path} then propogating rigid registration ({ax3dt1post_path} -> {rigid_template_path.split('/')[-1]}) onto each scan' >> {log_file}")
            for scan in swi_dwi_scans:
                cur_input_dir = f'{data_dir}/{subject}/{session}/{scan}'
                cur_output_dir = f'{output_dir}/{subject}/{session}/{scan}'
                scan_type = scan.split('-')[-1]
                if not os.path.exists(f'{cur_output_dir}/{session}_{scan}.nii.gz'): 
                    os.system(f"echo '\t\tProcessing: {scan}' >> {log_file}")
                    if not os.path.exists(cur_output_dir): os.makedirs(cur_output_dir)
                    shutil.copy(f'{cur_input_dir}/{session}_{scan}.json', f'{cur_output_dir}/{session}_{scan}.json')
                    original_mri = read_example_mri(data_dir, subject, session, scan, ants=True, orientation='IAL')

                    if scan_type == 'AX_SWI':
                        intermediary = swi_intermediary
                        intermediary_transform = swi_intermediary_transform
                    else:
                        intermediary = dwi_intermediary
                        intermediary_transform = dwi_intermediary_transform

                    intermediary_path = [s for s in current_scans if s.endswith(intermediary)][0]

                    intermediary_dir = data_dir
                    if skullstrip_dir: intermediary_dir = skullstrip_dir
                    os.system(f"echo '\t\t\tSubstep 1/3: Nonlinear warping {scan} to {intermediary_path} (from {intermediary_dir})' >> {log_file}")
                    intermediary_mri = read_example_mri(intermediary_dir, subject, session, intermediary_path, ants=True, orientation='IAL')
                    
                    type_of_transform = 'SyNRA'
                    try:
                        nonlinear_int_transform = ants.registration(
                            fixed=intermediary_mri,
                            moving=original_mri,
                            type_of_transform=type_of_transform,
                            verbose=False
                        )
                        nonlinear_int_transform['warpedmovout'].to_file(f'{cur_output_dir}/{session}_{scan}_{type_of_transform}_registration_to_{intermediary}.nii.gz')

                        os.system(f"echo '\t\t\tSubstep 2/3: Propogating the nonlinear warp from {intermediary_path} -> {ax3dt1post_path} onto {scan}' >> {log_file}")
                        # propogate intermediary transform
                        nonlinear_transform = ants.apply_transforms(
                            fixed=ax3dt1post,
                            moving=nonlinear_int_transform['warpedmovout'],
                            transformlist=intermediary_transform,
                            verbose=False
                        )

                        nonlinear_transform.to_file(f'{cur_output_dir}/{session}_{scan}_{type_of_transform}_propogated_registration_using_transform_from_{intermediary_path}_to_{nonlinear_template}.nii.gz')

                        os.system(f"echo '\t\t\tSubstep 3/3: Propogating the rigid registration from {ax3dt1post_path} -> {rigid_template_path.split('/')[-1]} onto {scan}' >> {log_file}")
                        propogated_rigid_transform = ants.apply_transforms(
                            fixed=mni_template,
                            moving=nonlinear_transform,
                            transformlist=rigid_transform['fwdtransforms'],
                            verbose=False
                        )

                        propogated_rigid_transform.to_file(f'{cur_output_dir}/{session}_{scan}.nii.gz')

                    except Exception as e:
                        os.system(f"echo '\t\t\tError: {e}' >> {log_file}")
                        os.system(f"echo '\t\t\tFixed shape: {intermediary_mri.numpy().shape}' >> {log_file}")
                        os.system(f"echo '\t\t\tMoving shape: {original_mri.numpy().shape}' >> {log_file}")
                        os.system(f"echo '\t\t\tUnable to register {scan}\n' >> {log_file}")

        time_elapsed = time.time() - begin_time
        os.system(f"echo 'Completed registration for the session: {session}' >> {log_file}")
        hours, rem = divmod(time_elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        time_elapsed = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
        os.system(f"echo 'Elapsed time: {time_elapsed}\n' >> {log_file}")

subjects = lsdir(data_dir)
for subject in subjects:
    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.submit(register_subject, subject)

overall_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
overall_time_elapsed = time.time() - overall_begin_time
hours, rem = divmod(overall_time_elapsed, 3600)
minutes, seconds = divmod(rem, 60)
overall_time_elapsed = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
os.system(f"cat {log_dir}/* >> {overall_log_file}")
os.system(f"echo '\nCompleted registration for all subjects at {overall_end_time}' >> {overall_log_file}")
os.system(f"echo 'Total elapsed time: {overall_time_elapsed}\n' >> {overall_log_file}")
os.system(f"echo '{bar}\n' >> {overall_log_file}")