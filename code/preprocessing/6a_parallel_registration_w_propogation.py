# File: 6a_parallel_registration_w_propogation.py
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
#   1. Intra subject registration to a AX 3D T1 POST image: aligns all scans of a subject to their own AX 3D T1 POST scan. 
#   2. Affine registration to MNI template: aligns the scans using only rotation and translation. Uses the MNI ICBM 152 nonlinear atlas version 2009 as the template.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import setup, lsdir, read_example_mri
from datetime import datetime
import time
import shutil
import os
import ants
import logging
from tqdm import tqdm

#-------------------------#
#### 1. FILE WRANGLING ####
#-------------------------#
setup()

skullstrip_dir = 'data/preprocessing/output/4_SKULLSTRIPPED' # set this to None if you don't want to use skullstripped intermediary images for SWI and DWI scans
data_dir = 'data/preprocessing/output/5b_ZSCORE_NORMALIZED'
output_dir = 'data/preprocessing/output/6b_REGISTERED'
log_dir = f'{output_dir}/logfiles'
num_workers = 1

if not os.path.exists(output_dir): os.makedirs(output_dir)
if not os.path.exists(log_dir): os.makedirs(log_dir)

intra_subject_template = 'AX_3D_T1_POST'
mni_template = 'https://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_sym_09a_nifti.zip' # 'MNI ICBM 152 nonlinear atlas version 2009'
mni_template_zip = mni_template.split('/')[-1]
mni_template_dir = mni_template_zip.split('_nifti')[0]

mni_template_path = f'{output_dir}/{mni_template_dir}/mni_icbm152_t1_tal_nlin_sym_09a.nii'
if not os.path.exists(mni_template_path):
    os.system(f"cd {output_dir} && wget {mni_template} && unzip {mni_template_zip} && rm {mni_template_zip} && cd ../../../../")

mni_template = ants.image_read(mni_template_path, reorient='IAL')

def save_transforms(tx, output_path):
    for i, t in enumerate(tx):
        suffix = t.split('.')[-1]
        shutil.copy(t, f'{output_path}_transform_tx_{i}.{suffix}')

#-----------------------#
#### 2. REGISTRATION ####
#-----------------------#
def register_subject(subject):
    log_file = os.path.join(log_dir, f'{subject}-log.txt')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')
    
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
        has_adc = False
        if 'AX_3D_T1_POST' in scan_types: has_post = True
        if 'AX_SWI' in scan_types: has_swi = True
        if 'AX_DIFFUSION' in scan_types: has_dwi = True
        if 'AX_3D_T1_PRE' in scan_types: has_pre = True
        if 'SAG_3D_FLAIR' in scan_types: has_flair = True
        if 'SAG_3D_T2' in scan_types: has_t2 = True
        if 'AX_ADC' in scan_types: has_adc = True

        if not has_post:
            logging.info(f"Warning: No {intra_subject_template} scan found for {session}, therefore skipping session: {session}")
            continue
        else:
            logging.info(f"Starting registration for the session: {session}")

        pre_req_scans = current_scans
        pre_req_scans = [s for s in pre_req_scans if not s.endswith('AX_3D_T1_POST')]
        if has_swi: pre_req_scans = [s for s in pre_req_scans if not s.endswith('AX_SWI')]
        if has_dwi: pre_req_scans = [s for s in pre_req_scans if not s.endswith('AX_DIFFUSION')]
        if has_adc: pre_req_scans = [s for s in pre_req_scans if not s.endswith('AX_ADC')]

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

        intra_subject_template_scan_path = [s for s in current_scans if s.endswith(intra_subject_template)][0]
        intra_subject_template_scan = read_example_mri(data_dir, subject, session, intra_subject_template_scan_path, ants=True, orientation='IAL')
        if intra_subject_template_scan.spacing != (1.0, 1.0, 1.0):
            logging.info(f"\tWarning: {session}/{intra_subject_template_scan_path} does not have 1x1x1mm spacing. Instead it has: {intra_subject_template_scan.spacing}")
        
        swi_intermediary_transform = None
        dwi_intermediary_transform = None

        # This is the first step of registration: register a subject's AX 3D T1 POST image to the MNI template
        type_of_transform = 'Affine'
        logging.info(f"\tPerforming {type_of_transform.lower()} registration to {mni_template_path.split('/')[-1]} for scan {intra_subject_template}")
        mni_transform = ants.registration(
            fixed=mni_template,
            moving=intra_subject_template_scan,
            type_of_transform=type_of_transform,
            verbose=False
        )

        cur_output_dir = f'{output_dir}/{subject}/{session}/{intra_subject_template_scan_path}'
        if not os.path.exists(cur_output_dir): os.makedirs(cur_output_dir)
        mni_transform['warpedmovout'].to_file(f'{cur_output_dir}/{session}_{intra_subject_template_scan_path}.nii.gz')
        shutil.copy(f'{data_dir}/{subject}/{session}/{intra_subject_template_scan_path}/{session}_{intra_subject_template_scan_path}.json', f'{cur_output_dir}/{session}_{intra_subject_template_scan_path}.json')
        save_transforms(mni_transform['fwdtransforms'], f'{cur_output_dir}/{session}_{intra_subject_template_scan_path}_{type_of_transform}_to_MNI')

        logging.info(f"\tAffine transforming the pre-requisite scans {pre_req_scans} to {intra_subject_template_scan_path}, then propogating affine registration ({intra_subject_template_scan_path} -> {mni_template_path.split('/')[-1]}) onto each scan")
        for scan in pre_req_scans:
            cur_input_dir = f'{data_dir}/{subject}/{session}/{scan}'
            cur_output_dir = f'{output_dir}/{subject}/{session}/{scan}'
            scan_type = scan.split('-')[-1]
            if not os.path.exists(f'{cur_output_dir}/{session}_{scan}.nii.gz'): 
                logging.info(f"\t\tProcessing: {scan}")
                if not os.path.exists(cur_output_dir): os.makedirs(cur_output_dir)
                shutil.copy(f'{cur_input_dir}/{session}_{scan}.json', f'{cur_output_dir}/{session}_{scan}.json')
                original_mri = read_example_mri(data_dir, subject, session, scan, ants=True, orientation='IAL')

                try:
                    logging.info(f"\t\t\tAffine transforming {scan} to {intra_subject_template_scan_path}")
                    type_of_transform = 'Affine'
                    intra_subject_transform = ants.registration(
                        fixed=intra_subject_template_scan,
                        moving=original_mri,
                        type_of_transform=type_of_transform,
                        verbose=False
                    )
                    intra_subject_transform['warpedmovout'].to_file(f'{cur_output_dir}/{session}_{scan}_{type_of_transform}_registration_to_{intra_subject_template}.nii.gz')
                    save_transforms(intra_subject_transform['fwdtransforms'], f'{cur_output_dir}/{session}_{scan}_{type_of_transform}_to_{intra_subject_template}')
                    if scan_type == swi_intermediary: swi_intermediary_transform = intra_subject_transform['fwdtransforms']
                    if scan_type == dwi_intermediary: dwi_intermediary_transform = intra_subject_transform['fwdtransforms']

                    logging.info(f"\t\t\tPropogating the affine registration from {intra_subject_template_scan_path} -> {mni_template_path.split('/')[-1]} onto {scan}")
                    propogated_mni_transform = ants.apply_transforms(
                        fixed=mni_template,
                        moving=intra_subject_transform['warpedmovout'],
                        transformlist=mni_transform['fwdtransforms'],
                        verbose=False
                    )

                    propogated_mni_transform.to_file(f'{cur_output_dir}/{session}_{scan}.nii.gz')

                except Exception as e:
                    logging.info(f"\t\t\tError: {e}")
                    logging.info(f"\t\t\tFixed shape: {intra_subject_template_scan.numpy().shape}")
                    logging.info(f"\t\t\tMoving shape: {original_mri.numpy().shape}")
                    logging.info(f"\t\t\tUnable to register {scan}\n")
        
        swi_dwi_scans = []
        if has_swi:
            swi_path = [s for s in current_scans if s.endswith('AX_SWI')][0]
            swi_dwi_scans.append(swi_path)
        if has_dwi:
            dwi_path = [s for s in current_scans if s.endswith('AX_DIFFUSION')][0]
            swi_dwi_scans.append(dwi_path)
        if has_adc:
            adc_path = [s for s in current_scans if s.endswith('AX_ADC')][0]
            swi_dwi_scans.append(adc_path)
        
        dwi_transform = None

        if len(swi_dwi_scans) > 0:
            logging.info(f"\tAffine transforming {swi_dwi_scans} scans to {intra_subject_template_scan_path} then propogating affine registration ({intra_subject_template_scan_path} -> {mni_template_path.split('/')[-1]}) onto each scan")
            for scan in swi_dwi_scans:
                cur_input_dir = f'{data_dir}/{subject}/{session}/{scan}'
                cur_output_dir = f'{output_dir}/{subject}/{session}/{scan}'
                scan_type = scan.split('-')[-1]
                if not os.path.exists(f'{cur_output_dir}/{session}_{scan}.nii.gz'): 
                    logging.info(f"\t\tProcessing: {scan}")
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
                    logging.info(f"\t\t\tSubstep 1/3: Affine transforming {scan} to {intermediary_path} (from {intermediary_dir})")
                    intermediary_mri = read_example_mri(intermediary_dir, subject, session, intermediary_path, ants=True, orientation='IAL')
                    
                    type_of_transform = 'Affine'
                    try:
                        if scan_type != 'AX_ADC':
                            intra_subject_int_transform = ants.registration(
                                fixed=intermediary_mri,
                                moving=original_mri,
                                type_of_transform=type_of_transform,
                                verbose=False
                            )
                            intra_subject_int_transform['warpedmovout'].to_file(f'{cur_output_dir}/{session}_{scan}_{type_of_transform}_registration_to_{intermediary}.nii.gz')
                            save_transforms(intra_subject_int_transform['fwdtransforms'], f'{cur_output_dir}/{session}_{scan}_{type_of_transform}_to_{intermediary}')
                            if scan_type == 'AX_DIFFUSION': dwi_transform = intra_subject_int_transform['fwdtransforms']
                        else:
                            intra_subject_int_transform = ants.apply_transforms(
                                fixed=intermediary_mri,
                                moving=original_mri,
                                transformlist=dwi_transform,
                                verbose=False
                            )
                            intra_subject_int_transform.to_file(f'{cur_output_dir}/{session}_{scan}_{type_of_transform}_registration_to_{intermediary}.nii.gz')
                        logging.info(f"\t\t\tSubstep 2/3: Propogating the affine transform from {intermediary_path} -> {intra_subject_template_scan_path} onto {scan}")
                        if scan_type != 'AX_ADC': intra_subject_int_transform = intra_subject_int_transform['warpedmovout']

                        intra_subject_transform = ants.apply_transforms(
                            fixed=intra_subject_template_scan,
                            moving=intra_subject_int_transform,
                            transformlist=intermediary_transform,
                            verbose=False
                        )

                        intra_subject_transform.to_file(f'{cur_output_dir}/{session}_{scan}_{type_of_transform}_propogated_registration_using_transform_from_{intermediary_path}_to_{intra_subject_template}.nii.gz')

                        logging.info(f"\t\t\tSubstep 3/3: Propogating the affine registration from {intra_subject_template_scan_path} -> {mni_template_path.split('/')[-1]} onto {scan}")
                        propogated_mni_transform = ants.apply_transforms(
                            fixed=mni_template,
                            moving=intra_subject_transform,
                            transformlist=mni_transform['fwdtransforms'],
                            verbose=False
                        )

                        propogated_mni_transform.to_file(f'{cur_output_dir}/{session}_{scan}.nii.gz')

                    except Exception as e:
                        logging.info(f"\t\t\tError: {e}")
                        logging.info(f"\t\t\tFixed shape: {intermediary_mri.numpy().shape}")
                        logging.info(f"\t\t\tMoving shape: {original_mri.numpy().shape}")
                        logging.info(f"\t\t\tUnable to register {scan}\n")

        time_elapsed = time.time() - begin_time
        logging.info(f"Completed registration for the session: {session}")
        hours, rem = divmod(time_elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        time_elapsed = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
        logging.info(f"Elapsed time: {time_elapsed}\n")

def main():
    # start overall logging
    overall_log_file = os.path.join(output_dir, 'log.txt')
    print(f"Logging output to {overall_log_file}")
    overall_begin_time = time.time()
    overall_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bar = '-' * 80
    os.system(f"echo '\n{bar}\n' >> {overall_log_file}")
    os.system(f"echo 'Running script 6a_parallel_registration_w_propogation.py at {overall_start_time}\n' >> {overall_log_file}")
    os.system(f"echo 'Skull stripping directory used: {skullstrip_dir}' >> {overall_log_file}")
    os.system(f"echo 'Intrasubject template used: {intra_subject_template}' >> {overall_log_file}")
    os.system(f"echo 'Affine template used: {mni_template_path}\n' >> {overall_log_file}")

    # register all subjects!!
    subjects = lsdir(data_dir)
    # reverse order of subjects
    subjects = subjects[::-1]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Create a list of futures
        futures = [executor.submit(register_subject, subject) for subject in subjects]
        
        # Initialize tqdm progress bar
        with tqdm(total=len(futures)) as progress_bar:
            for _ in as_completed(futures):
                progress_bar.update(1)
                # Result of the future can be obtained if needed
                # result = future.result()

    # end overall logging
    overall_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    overall_time_elapsed = time.time() - overall_begin_time
    hours, rem = divmod(overall_time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    overall_time_elapsed = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
    os.system(f"cat {log_dir}/* >> {overall_log_file}")
    os.system(f"echo '\nCompleted registration for all subjects at {overall_end_time}' >> {overall_log_file}")
    os.system(f"echo 'Total elapsed time: {overall_time_elapsed}\n' >> {overall_log_file}")
    os.system(f"echo '{bar}\n' >> {overall_log_file}")

if __name__ == '__main__':
    main()