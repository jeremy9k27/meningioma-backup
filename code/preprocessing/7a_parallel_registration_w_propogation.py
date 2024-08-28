# File: 7a_parallel_registration_w_propogation.py
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

tx_dir = f'data/preprocessing/output/6b_REGISTERED' # directory where the transforms are saved to propogate onto new AX_ADC scans
skullstrip_dir = 'data/round2_preprocessing/output/5_SKULLSTRIPPED' # set this to None if you don't want to use skullstripped intermediary images for SWI and DWI scans
data_dir = 'data/round2_preprocessing/output/6_ZSCORE_NORMALIZED'
output_dir = 'data/round2_preprocessing/output/7_REGISTERED'
log_dir = f'{output_dir}/logfiles'
num_workers = 4

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
    try:
        for i, t in enumerate(tx):
            suffix = '.'.join(t.split('.')[1:])
            logging.info(f"\t\t\tAttempting to save transform {i}/{len(tx)} as {output_path}_transform_tx_{i}.{suffix}...")
            shutil.copy(t, f'{output_path}_transform_tx_{i}.{suffix}')
            logging.info(f"\t\t\tSuccess!")
    except Exception as e:
        logging.info(f"\t\t\tError in saving transforms: {e}")

#-----------------------#
#### 2. REGISTRATION ####
#-----------------------#
def prop_existing_txs(subject):
    # Setting up logging
    log_file = os.path.join(log_dir, f'{subject}-log.txt')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')
    begin_time = time.time()

    # File wrangling
    session = lsdir(f'{tx_dir}/{subject}')[0]
    current_scans = lsdir(f'{tx_dir}/{subject}/{session}')
    scan_types = [scan.split('-')[-1] for scan in current_scans]
    
    # If there is no AX 3D T1 POST scan, then we can't register the session, since that is the intra-subject template
    post_path = [s for s in current_scans if s.endswith(intra_subject_template)]
    if len(post_path) == 0:
        logging.info(f"Warning: No {intra_subject_template} scan found for {session}, therefore skipping session: {session}")
        return
    else:
        logging.info(f"Starting registration for the session: {session}")
        post_path = post_path[0]

    # Obtain all relevant scans for the session
    adc_path = [s for s in current_scans if s.endswith('AX_ADC')][0] # there must be an ADC scan for this fn to have been called, so we can index it safely here
    dwi_path = [s for s in current_scans if s.endswith('AX_DIFFUSION')]
    dwi_path = dwi_path[0] if len(dwi_path) > 0 else None
    flair_path = [s for s in current_scans if s.endswith('SAG_3D_FLAIR')]
    flair_path = flair_path[0] if len(flair_path) > 0 else None
    pre_path = [s for s in current_scans if s.endswith('AX_3D_T1_PRE')]
    pre_path = pre_path[0] if len(pre_path) > 0 else None
    t2_path = [s for s in current_scans if s.endswith('SAG_3D_T2')]
    t2_path = t2_path[0] if len(t2_path) > 0 else None

    # Determine which intermediary scan to use
    dwi_intermediary = None
    if dwi_path and flair_path:
        dwi_intermediary = 'SAG_3D_FLAIR'
    elif dwi_path and pre_path:
        dwi_intermediary = 'AX_3D_T1_PRE'
    elif dwi_path and t2_path:
        dwi_intermediary = 'SAG_3D_T2'
    else:
        logging.info(f"Warning: No intermediary scan found for {session}, therefore skipping session: {session}")
        return
    intermediary_path = [s for s in current_scans if s.endswith(dwi_intermediary)][0]

    # Read in current ADC scan
    adc_scan = read_example_mri(data_dir, subject, session, adc_path, ants=True, orientation='IAL')

    # Read in the existing transforms
    #### PROPOGATION 1: ADC -> dwi_intermediary e.g. SAG 3D FLAIR (via DWI) ####
    tx1_path = f'{tx_dir}/{subject}/{session}/{dwi_path}/{session}_{dwi_path}_Affine_to_{dwi_intermediary}_transform_tx_0.mat'
    #### PROPOGATION 2: dwi_intermediary e.g. SAG 3D FLAIR -> AX 3D T1 POST ####
    tx2_path = f'{tx_dir}/{subject}/{session}/{intermediary_path}/{session}_{intermediary_path}_Affine_to_{intra_subject_template}_transform_tx_0.mat'
    #### PROPOGATION 3: AX 3D T1 POST -> MNI TEMPLATE ####
    tx3_path = f'{tx_dir}/{subject}/{session}/{post_path}/{session}_{post_path}_Affine_to_MNI_transform_tx_0.mat'

    # Propogate the transforms onto the new ADC scan
    logging.info(f"\tPropogating existing transforms onto new ADC scan: {adc_path}")
    try:
        propogated_mni_transform = ants.apply_transforms(
            fixed=mni_template,
            moving=adc_scan,
            transformlist=[tx1_path, tx2_path, tx3_path],
            verbose=False
        )
        if not os.path.exists(f'{output_dir}/{subject}/{session}/{adc_path}'): os.makedirs(f'{output_dir}/{subject}/{session}/{adc_path}')
        propogated_mni_transform.to_file(f'{output_dir}/{subject}/{session}/{adc_path}/{session}_{adc_path}.nii.gz')
        logging.info(f"\tSuccessfully registered {adc_path}\n")
    except Exception as e:
        logging.info(f"\tError: {e}")
        logging.info(f"\tUnable to register {adc_path}\n")
    
    # Log the completion of the registration
    time_elapsed = time.time() - begin_time
    logging.info(f"Completed registration for the session: {session}")
    hours, rem = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    time_elapsed = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
    logging.info(f"Elapsed time: {time_elapsed}\n")

def register_subject(subject):
    """
    Registration is done as follows:
    1. Register the AX 3D T1 POST scan to the MNI template, save the resulting transforms
    2. Register the 'pre-requisite' scans (e.g. SAG 3D FLAIR) to the AX 3D T1 POST scan, save the resulting transforms
    3. Propogate the transforms from the AX 3D T1 POST -> MNI template onto the pre-requisite scans
    4. Register SWI, and DWI scans to the appropriate pre-requisite scan, save the resulting transforms
    5. Propogate the transforms from the DWI -> pre-req scan onto the ADC scan
    6. Propogate the transforms from the pre-req scan -> AX 3D T1 POST onto the SWI, DWI, and ADC scans
    7. Propogate the transforms from the AX 3D T1 POST -> MNI template onto the SWI, DWI, and ADC scans
    In this way, all scans are registered to the MNI template, and the transforms are saved for each step.

    * E.g., ADC -> SAG 3D FLAIR (via DWI) -> AX 3D T1 POST -> MNI template
    * E.g., DWI, SWI -> SAG 3D FLAIR -> AX 3D T1 POST -> MNI template
    * E.g., SAG 3D FLAIR -> AX 3D T1 POST -> MNI template
    * E.g., AX 3D T1 PRE -> AX 3D T1 POST -> MNI template
    * E.g., AX 3D T1 POST -> MNI template
    """
    # Some of the subjects have already been registered from the previous round of pre-processing
    # But, we have rescaled their AX_ADC scans since that is a global normalization step, so we need to re-register them
    # We probably don't want to re-register them from scratch, 1) because the work has already been done, 
    # and 2) because those subjects were segmented with those original registrations, and we definitely don't want to re-segment them
    # So, we will propogate the existing transforms onto the new AX_ADC scans in a separate function
    if subject not in lsdir(skullstrip_dir):
        prop_existing_txs(subject)
        return
    
    # Setting up logging
    log_file = os.path.join(log_dir, f'{subject}-log.txt')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')
    begin_time = time.time()

    # Go thru each session for the subject, one at a time
    for session in lsdir(f'{data_dir}/{subject}'):

        # Read in available scans for the session
        current_scans = lsdir(f'{data_dir}/{subject}/{session}')
        scan_types = [scan.split('-')[-1] for scan in current_scans]

        # Keep track of which scans are available for the session
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

        # If there is no AX 3D T1 POST scan, then there is no intra-subject template, so we must simply register all scans directly to the MNI template
        #### REGISTRATION 1: ALL SCANS -> MNI TEMPLATE if there is no T1 POST ####
        if not has_post:
            logging.info(f"Warning: No {intra_subject_template} scan found for {session}, therefore registering all scans in the session straight to the MNI template")
            for scan in current_scans:
                cur_input_dir = f'{data_dir}/{subject}/{session}/{scan}'
                cur_output_dir = f'{output_dir}/{subject}/{session}/{scan}'
                scan_type = scan.split('-')[-1]
                if not os.path.exists(f'{cur_output_dir}/{session}_{scan}.nii.gz'): 
                    logging.info(f"\tProcessing: {scan}")
                    if not os.path.exists(cur_output_dir): os.makedirs(cur_output_dir)
                    shutil.copy(f'{cur_input_dir}/{session}_{scan}.json', f'{cur_output_dir}/{session}_{scan}.json')
                    original_mri = read_example_mri(data_dir, subject, session, scan, ants=True, orientation='IAL')
                    try:
                        # Register the scan to the MNI template
                        type_of_transform = 'Affine'
                        mni_transform = ants.registration(
                            fixed=mni_template,
                            moving=original_mri,
                            type_of_transform=type_of_transform,
                            verbose=False
                        )
                        # Save the registered image
                        mni_transform['warpedmovout'].to_file(f'{cur_output_dir}/{session}_{scan}.nii.gz')
                        # Save the actual transforms
                        save_transforms(mni_transform['fwdtransforms'], f'{cur_output_dir}/{session}_{scan}_{type_of_transform}_to_MNI')
                    except Exception as e:
                        logging.info(f"\tError: {e}")
                        logging.info(f"\tUnable to register {scan}\n")
            continue
        else:
            logging.info(f"Starting registration for the session: {session}")

        # Pre-requisite scans are those scans that can be registered to the intra-subject template (AX 3D T1 POST), 
        # and then propogated to the MNI template, without any intermediary steps (e.g., SAG 3D FLAIR, AX 3D T1 PRE, SAG 3D T2).
        # Other scans like SWI, DIFFUSION, ADC, then rely on the saved transforms from the pre-requisite scans, 
        # since they require an intermediary step, e.g., AX_ADC -> SAG 3D FLAIR -> AX T1 POST -> MNI template 
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

        #### REGISTRATION 1: AX 3D T1 POST -> MNI TEMPLATE ####
        # Get the intra-subject template scan for the session (AX 3D T1 POST), and make sure it has 1x1x1mm spacing
        intra_subject_template_scan_path = [s for s in current_scans if s.endswith(intra_subject_template)][0]
        intra_subject_template_scan = read_example_mri(data_dir, subject, session, intra_subject_template_scan_path, ants=True, orientation='IAL')
        if intra_subject_template_scan.spacing != (1.0, 1.0, 1.0):
            logging.info(f"\tWarning: {session}/{intra_subject_template_scan_path} does not have 1x1x1mm spacing. Instead it has: {intra_subject_template_scan.spacing}")
        
        # This is the first step of registration: register a subject's AX 3D T1 POST image to the MNI template
        type_of_transform = 'Affine'
        logging.info(f"\tPerforming {type_of_transform.lower()} registration to {mni_template_path.split('/')[-1]} for scan {intra_subject_template}")
        mni_transform = ants.registration(
            fixed=mni_template,
            moving=intra_subject_template_scan,
            type_of_transform=type_of_transform,
            verbose=False
        )

        # Save the registered AX 3D T1 POST image (registered to MNI template) and the transforms from that registration
        cur_output_dir = f'{output_dir}/{subject}/{session}/{intra_subject_template_scan_path}'
        if not os.path.exists(cur_output_dir): os.makedirs(cur_output_dir)
        mni_transform['warpedmovout'].to_file(f'{cur_output_dir}/{session}_{intra_subject_template_scan_path}.nii.gz')
        shutil.copy(f'{data_dir}/{subject}/{session}/{intra_subject_template_scan_path}/{session}_{intra_subject_template_scan_path}.json', f'{cur_output_dir}/{session}_{intra_subject_template_scan_path}.json')
        save_transforms(mni_transform['fwdtransforms'], f'{cur_output_dir}/{session}_{intra_subject_template_scan_path}_{type_of_transform}_to_MNI')

        # Initialize intermediary transforms for SWI and DWI scans
        swi_intermediary_transform = None
        dwi_intermediary_transform = None

        #### REGISTRATIONS 2 & 3: PRE-REQUISITE SCANS -> AX 3D T1 POST -> MNI TEMPLATE ####
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
                    #### REGISTRATION 2: PRE-REQUISITE SCANS -> AX 3D T1 POST ####
                    logging.info(f"\t\t\tAffine transforming {scan} to {intra_subject_template_scan_path}")
                    type_of_transform = 'Affine'
                    intra_subject_transform = ants.registration(
                        fixed=intra_subject_template_scan,
                        moving=original_mri,
                        type_of_transform=type_of_transform,
                        verbose=False
                    )
                    # Save transformed image
                    intra_subject_transform['warpedmovout'].to_file(f'{cur_output_dir}/{session}_{scan}_{type_of_transform}_registration_to_{intra_subject_template}.nii.gz')
                    # Save the actual transforms
                    save_transforms(intra_subject_transform['fwdtransforms'], f'{cur_output_dir}/{session}_{scan}_{type_of_transform}_to_{intra_subject_template}')
                    if scan_type == swi_intermediary: swi_intermediary_transform = intra_subject_transform['fwdtransforms']
                    if scan_type == dwi_intermediary: dwi_intermediary_transform = intra_subject_transform['fwdtransforms']

                    #### REGISTRATION 3: PROPOGATE AX 3D T1 POST -> MNI template onto PRE-REQ SCANS ####
                    logging.info(f"\t\t\tPropogating the registration from {intra_subject_template_scan_path} -> {mni_template_path.split('/')[-1]} onto {scan}")
                    propogated_mni_transform = ants.apply_transforms(
                        fixed=mni_template,
                        moving=intra_subject_transform['warpedmovout'],
                        transformlist=mni_transform['fwdtransforms'],
                        verbose=False
                    )
                    # Save the finalized registered PRE REQ image
                    propogated_mni_transform.to_file(f'{cur_output_dir}/{session}_{scan}.nii.gz')

                except Exception as e:
                    logging.info(f"\t\t\tError: {e}")
                    logging.info(f"\t\t\tFixed shape: {intra_subject_template_scan.numpy().shape}")
                    logging.info(f"\t\t\tMoving shape: {original_mri.numpy().shape}")
                    logging.info(f"\t\t\tUnable to register {scan}\n")
        
        # Next step: dealing with SWI, DWI, and ADC scans
        # Make a list of the scans needing registration
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
        
        # Initialize intermediary transform
        dwi_transform = None

        if len(swi_dwi_scans) > 0:
            logging.info(f"\tAffine transforming {swi_dwi_scans} scans to {intra_subject_template_scan_path} then propogating affine registration ({intra_subject_template_scan_path} -> {mni_template_path.split('/')[-1]}) onto each scan")
            for scan in swi_dwi_scans:
                # Set up filepaths for the current scan, read in the current scan
                cur_input_dir = f'{data_dir}/{subject}/{session}/{scan}'
                cur_output_dir = f'{output_dir}/{subject}/{session}/{scan}'
                scan_type = scan.split('-')[-1]
                if not os.path.exists(f'{cur_output_dir}/{session}_{scan}.nii.gz'): 
                    logging.info(f"\t\tProcessing: {scan}")
                    if not os.path.exists(cur_output_dir): os.makedirs(cur_output_dir)
                    shutil.copy(f'{cur_input_dir}/{session}_{scan}.json', f'{cur_output_dir}/{session}_{scan}.json')
                    original_mri = read_example_mri(data_dir, subject, session, scan, ants=True, orientation='IAL')

                    # Set up intermediary scan and transforms
                    if scan_type == 'AX_SWI':
                        intermediary = swi_intermediary
                        intermediary_transform = swi_intermediary_transform
                    else:
                        intermediary = dwi_intermediary
                        intermediary_transform = dwi_intermediary_transform

                    intermediary_path = [s for s in current_scans if s.endswith(intermediary)][0]

                    # For SWI, DWI, and ADC scans, we want to use the skullstripped images as intermediaries if they exist, otherwise use the original images
                    intermediary_dir = data_dir
                    if skullstrip_dir: intermediary_dir = skullstrip_dir
                    logging.info(f"\t\t\tSubstep 1/3: Affine transforming {scan} to {intermediary_path} (from {intermediary_dir})")
                    intermediary_mri = read_example_mri(intermediary_dir, subject, session, intermediary_path, ants=True, orientation='IAL')
                    
                    type_of_transform = 'Affine'
                    try:
                        # ADC is last in the list and originates from the Diffusion scan, so all we have to do first is propogate the transform from the Diffusion scan below
                        # For everything else, we first need to register the scan to the intermediary scan
                        if scan_type != 'AX_ADC':
                            #### REGISTRATION 4: SWI, DWI -> INTERMEDIARY SCAN ####
                            intra_subject_int_transform = ants.registration(
                                fixed=intermediary_mri,
                                moving=original_mri,
                                type_of_transform=type_of_transform,
                                verbose=False
                            )
                            # Save the transformed image
                            intra_subject_int_transform['warpedmovout'].to_file(f'{cur_output_dir}/{session}_{scan}_{type_of_transform}_registration_to_{intermediary}.nii.gz')
                            # Save the actual transforms
                            save_transforms(intra_subject_int_transform['fwdtransforms'], f'{cur_output_dir}/{session}_{scan}_{type_of_transform}_to_{intermediary}')
                            # Set the transform for the DWI scan to propogate onto the ADC scan below
                            if scan_type == 'AX_DIFFUSION': dwi_transform = intra_subject_int_transform['fwdtransforms']
                        else:
                            #### REGISTRATION 5: ADC -> DWI SCAN ####
                            intra_subject_int_transform = ants.apply_transforms(
                                fixed=intermediary_mri,
                                moving=original_mri,
                                transformlist=dwi_transform,
                                verbose=False
                            )
                            # Save the transformed image
                            intra_subject_int_transform.to_file(f'{cur_output_dir}/{session}_{scan}_{type_of_transform}_registration_to_{intermediary}.nii.gz')
                        logging.info(f"\t\t\tSubstep 2/3: Propogating the affine transform from {intermediary_path} -> {intra_subject_template_scan_path} onto {scan}")
                        # Only use the warpedmovout if the scan is not an ADC scan, since if it is the ADC scan, the intra_subject_int_transform is already the warpedmovout (since it was propogated from the DWI scan)
                        if scan_type != 'AX_ADC': intra_subject_int_transform = intra_subject_int_transform['warpedmovout']

                        #### REGISTRATION 6: PROPOGATE INTERMEDIARY SCAN -> AX 3D T1 POST transform onto SWI, DWI, ADC ####
                        intra_subject_transform = ants.apply_transforms(
                            fixed=intra_subject_template_scan,
                            moving=intra_subject_int_transform,
                            transformlist=intermediary_transform,
                            verbose=False
                        )
                        intra_subject_transform.to_file(f'{cur_output_dir}/{session}_{scan}_{type_of_transform}_propogated_registration_using_transform_from_{intermediary_path}_to_{intra_subject_template}.nii.gz')

                        #### REGISTRATION 7: PROPOGATE AX 3D T1 POST -> MNI template onto SWI, DWI, ADC ####
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
    os.system(f"echo 'Running script 7a_parallel_registration_w_propogation.py at {overall_start_time}\n' >> {overall_log_file}")
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

# %%
