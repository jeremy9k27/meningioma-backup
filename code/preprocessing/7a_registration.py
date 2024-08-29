import os
import ants
from utils import setup, lsdir
import logging
import shutil
import time
from datetime import datetime
from tqdm import tqdm

class Registration:
    def __init__(self):
        """
        Registration class to facilitate registration of scans to MNI template.
        """

        # Global directories
        self.prev_round_dir = f'data/preprocessing/output/6b_REGISTERED'
        self.skullstrip_dir = 'data/round2_preprocessing/output/5_SKULLSTRIPPED' # set this to None if you don't want to use skullstripped intermediary images for SWI and DWI scans
        self.data_dir = 'data/round2_preprocessing/output/6_ZSCORE_NORMALIZED'
        self.output_dir = 'data/round2_preprocessing/output/7_REGISTERED'
        self.log_dir = f'{self.output_dir}/logfiles'

        # Create directories if they don't exist
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)
        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)

        # Setting up MNI template files
        self.intra_subject_template = 'AX_3D_T1_POST'
        self.mni_template = 'https://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_sym_09a_nifti.zip' # 'MNI ICBM 152 nonlinear atlas version 2009'
        self.mni_template_zip = self.mni_template.split('/')[-1]
        self.mni_template_dir = self.mni_template_zip.split('_nifti')[0]
        self.mni_template_path = f'{self.output_dir}/{self.mni_template_dir}/mni_icbm152_t1_tal_nlin_sym_09a.nii'
        if not os.path.exists(self.mni_template_path):
            os.system(f"cd {self.output_dir} && wget {self.mni_template} && unzip {self.mni_template_zip} && rm {self.mni_template_zip} && cd ../../../../")
        self.mni_template_ants = ants.image_read(self.mni_template_path, reorient='IAL')

        # Defining the kind of registration
        self.tx_type = 'Affine'

        # Defining the scans that need to be registered using an intermediary
        self.scans_needing_intermediary = ['AX_SWI', 'AX_DIFFUSION', 'AX_ADC']

        # Setting up main logging
        self.main_logger = logging.getLogger('main_logger')
        self.main_logger.setLevel(logging.INFO)
        self.main_handler = logging.FileHandler(f'{self.output_dir}/log.txt')
        self.main_handler.setLevel(logging.INFO)
        self.main_formatter = logging.Formatter('%(message)s')
        self.main_handler.setFormatter(self.main_formatter)
        self.main_logger.addHandler(self.main_handler)

        # Initializing the logger for the current registration
        self.current_logger = None
        self.current_handler = None
        self.current_formatter = None

        # Logging the settings for this registration
        self.bar = '-' * 80
        self.start_time = time.time()
        self.start_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.main_logger.info(f"{self.bar}")
        self.main_logger.info(f"Initializing Registration at: {self.start_date}")
        self.main_logger.info(f"Registration settings:")
        self.main_logger.info(f"\tprev_round_dir: {self.prev_round_dir}")
        self.main_logger.info(f"\tskullstrip_dir: {self.skullstrip_dir}")
        self.main_logger.info(f"\tdata_dir: {self.data_dir}")
        self.main_logger.info(f"\toutput_dir: {self.output_dir}")
        self.main_logger.info(f"\tlog_dir: {self.log_dir}")
        self.main_logger.info(f"\tintra_subject_template: {self.intra_subject_template}")
        self.main_logger.info(f"\tmni_template: {self.mni_template}")
        self.main_logger.info(f"\tmni_template_path: {self.mni_template_path}")
        self.main_logger.info(f"\ttx_type: {self.tx_type}")
        self.main_logger.info(f"\tscans_needing_intermediary: {self.scans_needing_intermediary}")
        self.main_logger.info(f"{self.bar}")

        # Keeping track of the failed registrations
        self.num_failed_registrations = 0
        self.failed_registrations = []

        # Subject-session-specific information
        self.subject = None
        self.session = None
        self.scan_available = { # Note, the order here is used to determine the order of registration
            'AX_3D_T1_POST': False,
            'SAG_3D_FLAIR': False,
            'AX_3D_T1_PRE': False,
            'SAG_3D_T2': False,
            'AX_DIFFUSION': False,
            'AX_SWI': False,
            'AX_ADC': False,
            'AX_2D_T2': False,
        }
        self.scan_paths = None
    
    def _end_logging(self):
        # Calculating the time elapsed
        end_time = time.time()
        end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hours, rem = divmod(end_time - self.start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        overall_time_elapsed = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

        # Logging the end of the registration
        self.main_logger.info(f"{self.bar}")
        self.main_logger.info(f"Registration completed at: {end_date}")
        self.main_logger.info(f"Total time elapsed: {overall_time_elapsed}")
        self.main_logger.info(f"Number of failed registrations: {self.num_failed_registrations}")
        self.main_logger.info(f"Failed registrations:")
        for failed_registration in self.failed_registrations:
            self.main_logger.info(f"\t{failed_registration}")
        self.main_logger.info(f"{self.bar}")
        self.main_handler.close()
        self.main_logger.removeHandler(self.main_handler)
    
    def _reset_session_info(self):
        self.session = None
        self.scan_available = {
            'AX_3D_T1_POST': False,
            'SAG_3D_FLAIR': False,
            'AX_3D_T1_PRE': False,
            'SAG_3D_T2': False,
            'AX_DIFFUSION': False,
            'AX_SWI': False,
            'AX_ADC': False,
            'AX_2D_T2': False,
            'SAG_3D_T2': False,
        }
        self.scan_paths = None

    def _get_scan_path(self, scan_type):
        assert self.scan_paths is not None, "The scan_paths attribute must be set before calling this method."
        scan_path = [path for path in self.scan_paths if path.endswith(scan_type)]
        assert len(scan_path) == 1, f"Found {len(scan_path)} paths for scan type {scan_type}."
        return scan_path[0]
    
    def _save_transforms(self, tx, output_path):
        for i, t in enumerate(tx):
            suffix = '.'.join(t.split('.')[1:])
            shutil.copy(t, f'{output_path}_{i}.{suffix}')

    def _direct_register(self, fixed_path, moving_path, output_path):
        """Registers the moving path to the fixed path, saving the result to output_path and the transforms via _save_transforms."""
        fixed_im = ants.image_read(fixed_path, reorient='IAL')
        moving_im = ants.image_read(moving_path, reorient='IAL')
        result = ants.registration(fixed=fixed_im, moving=moving_im, type_of_transform=self.tx_type, verbose=False)
        result['warpedmovout'].to_file(output_path)
        tx_dir = '/'.join(output_path.split('/')[:-1])
        if 'mni' in fixed_path:
            fixed_name = "MNI"
        else:
            fixed_name = fixed_path.split('/')[-1].split('.')[0].split('-')[-1]
        moving_name = moving_path.split('/')[-1].split('.')[0].split('-')[-1]
        tx_path = f'{tx_dir}/TX-{moving_name}_to_{fixed_name}_tx' # e.g. 'TX-AX_3D_T1_POST_to_MNI_tx' or 'TX-SAG_3D_FLAIR_to_AX_3D_T1_POST_tx'
        self._save_transforms(result['fwdtransforms'], tx_path)
        return result['warpedmovout']
    
    def _propagate_register(self, moving_im, tx_list, output_path):
        result = ants.apply_transforms(fixed=self.mni_template_ants, moving=moving_im, transformlist=tx_list, verbose=False)
        result.to_file(output_path)
    
    def _get_intermediary_scan(self):
        """Get the intermediary scan path, if it exists."""
        if self.scan_available['SAG_3D_FLAIR']:
            return self._get_scan_path('SAG_3D_FLAIR')
        elif self.scan_available['AX_3D_T1_PRE']:
            return self._get_scan_path('AX_3D_T1_PRE')
        elif self.scan_available['SAG_3D_T2']:
            return self._get_scan_path('SAG_3D_T2')
        else:
            return None
    
    def _unravel_available_txs(self, scan_type):
        next_scan = scan_type
        all_tx_files = []
        while next_scan != 'MNI':
            scan_path = self._get_scan_path(next_scan)
            cur_dir = f'{self.output_dir}/{self.subject}/{self.session}/{scan_path}'
            cur_files = os.listdir(cur_dir)
            tx_files = [f"{cur_dir}/{cf}" for cf in cur_files if cf.startswith('TX-')]
            all_tx_files.extend(tx_files)
            next_scan = tx_files[0].split('/')[-1].split('_to_')[-1].split('_tx_')[0]
        return all_tx_files
    
    def _register_scan(self, scan_type):
        # Obtain the current scan path to be registered
        scan_path = self._get_scan_path(scan_type) # e.g. '10-AX_3D_T1_POST' or # '3-AX_DIFFUSION'
        scan_path_full = f'{self.data_dir}/{self.subject}/{self.session}/{scan_path}/{self.session}_{scan_path}.nii.gz'
        cur_output_dir = f'{self.output_dir}/{self.subject}/{self.session}/{scan_path}'
        if not os.path.exists(cur_output_dir): os.makedirs(cur_output_dir)
        shutil.copy(
            f'{self.data_dir}/{self.subject}/{self.session}/{scan_path}/{self.session}_{scan_path}.json',
            f'{cur_output_dir}/{self.session}_{scan_path}.json'
        )
        cur_output_path = f'{cur_output_dir}/{self.session}_{scan_path}.nii.gz'

        # Setting up the logger for the current registration
        self.current_logger = logging.getLogger(f'{self.session}_{scan_path}_logger')
        self.current_logger.setLevel(logging.INFO)
        self.current_handler = logging.FileHandler(f'{self.log_dir}/{self.session}_{scan_path}.txt')
        self.current_handler.setLevel(logging.INFO)
        self.current_formatter = logging.Formatter('%(message)s')
        self.current_handler.setFormatter(self.current_formatter)
        self.current_logger.addHandler(self.current_handler)

        # Logging the start of the registration
        start_time = time.time()
        start_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.current_logger.info(f"{self.bar}")
        self.current_logger.info(f"{self.session}_{scan_path} Logfile")
        self.current_logger.info(f"\tStart date: {start_date}")
        self.current_logger.info(f"\tScan type: {scan_type}")

        #### REGISTRATION ####
        try:
            if scan_type == self.intra_subject_template:
                # In this case, we want to register scan_path to the MNI template directly
                self.current_logger.info(f"\tDirect registering {scan_path} to the MNI template...")
                
                # Read in intra subject template and check it's spacing
                moving_im = ants.image_read(f"{scan_path_full}", reorient='IAL')
                if moving_im.spacing != (1.0, 1.0, 1.0):
                    self.current_logger.info(f"\tWarning: {scan_path} does not have 1x1x1mm spacing, but rather {moving_im.spacing} spacing. Proceeding w/registration...")
                    # moving_im = ants.resample_image(moving_im, (1.0, 1.0, 1.0), use_voxels=True)
                
                # 1/1: Register the scan to the MNI template directly (the transforms are saved in _direct_register)
                self._direct_register(
                    fixed_path=self.mni_template_path, 
                    moving_path=scan_path_full, 
                    output_path=cur_output_path
                )
                self.current_logger.info(f"\tRegistration successfully completed.")

            elif scan_type in self.scans_needing_intermediary:
                # In this case, we want to: 
                # 1) register scan_path to the appropriate intermediary scan (if available), then;
                # 2) propagate the intermediary scan's registration to the intra_subject_template onto scan_path, and finally;
                # 3) propagate the intra_subject_template's registration to the MNI template onto scan_path.
                # Should any of these steps be missing, we simply skip the step and continue to the next step.
                # We also need to pay special attention to the case that this is an ADC scan, in which case step 1) is replaced
                # with the following:
                # 1*) Propagate the registration of the DWI scan to the intermediary/intra_subject_template/mni template 
                #     (whichever is available) onto the ADC scan. 
                # Finally, we want to use the skullstripped directory for these registrations, if available.
                intermediary_origin_dir = self.skullstrip_dir if self.skullstrip_dir else self.data_dir
                if scan_type == 'AX_ADC':
                    # Obtain intermediary scan incase we need it
                    int_scan = self._get_intermediary_scan()

                    # Ideally we have the AX_DIFFUSION scan available and can just propagate it's entire registration history onto the ADC.
                    if self.scan_available['AX_DIFFUSION']: 
                        # 1*/1: propagate the AX_DIFFUSION scan's entire registration history to the ADC scan
                        self.current_logger.info(f"\tStep 1/1: Propagating AX_DIFFUSION scan's entire registration history to the AX_ADC scan...")
                        available_txs = self._unravel_available_txs('AX_DIFFUSION')
                        self.current_logger.info(f"\t\tAvailable transforms: {available_txs}")
                        self._propagate_register(
                            moving_im=ants.image_read(scan_path_full, reorient='IAL'), 
                            tx_list=available_txs,
                            output_path=cur_output_path
                        )
                        self.current_logger.info(f"\tRegistration successfully completed.")
                    elif int_scan:
                        # 1/2: register scan_path to the appropriate intermediary scan
                        self.current_logger.info(f"\tStep 1/2: Registering {scan_path} to the intermediary scan {int_scan}...")
                        int_scan_path_full = f'{intermediary_origin_dir}/{self.subject}/{self.session}/{int_scan}/{self.session}_{int_scan}.nii.gz'
                        step1 = self._direct_register(
                            fixed_path=int_scan_path_full,
                            moving_path=scan_path_full,
                            output_path=f'{cur_output_dir}/{self.session}_{scan_path}_direct_reg_to_{int_scan}_.nii.gz'
                        )
                        # 2/2: unravel which existing transforms are available to propagate
                        available_txs = self._unravel_available_txs(int_scan.split('-')[-1])
                        self.current_logger.info(f"\tStep 2/2: Propagating available transforms onto {scan_path}...")
                        self.current_logger.info(f"\t\tAvailable transforms: {available_txs}")
                        self._propagate_register(
                            moving_im=step1,
                            tx_list=available_txs,
                            output_path=cur_output_path
                        )
                        self.current_logger.info(f"\tRegistration successfully completed.")
                    else:
                        # 1/1: register scan_path to the MNI template directly
                        self.current_logger.info(f"\tStep 1/1: Neither the intra subject template {self.intra_subject_template} nor an appropriate intermediary scan is available, therefore registering {scan_path} directly to MNI template...")
                        self._direct_register(
                            fixed_path=self.mni_template_path, 
                            moving_path=scan_path_full, 
                            output_path=cur_output_path
                        )
                        self.current_logger.info(f"\tRegistration successfully completed.")
                else:
                    intermediary_scan_path = self._get_intermediary_scan()
                    if intermediary_scan_path:
                        # 1/2: register scan_path to the appropriate intermediary scan
                        self.current_logger.info(f"\tStep 1/2: Registering {scan_path} to the intermediary scan {intermediary_scan_path}...")
                        intermediary_scan_path_full = f'{intermediary_origin_dir}/{self.subject}/{self.session}/{intermediary_scan_path}/{self.session}_{intermediary_scan_path}.nii.gz'
                        step1 = self._direct_register(
                            fixed_path=intermediary_scan_path_full,
                            moving_path=scan_path_full,
                            output_path=f'{cur_output_dir}/{self.session}_{scan_path}_direct_reg_to_{intermediary_scan_path}_.nii.gz'
                        )
                        # 2/2: unravel which existing transforms are available to propagate
                        available_txs = self._unravel_available_txs(scan_type=intermediary_scan_path.split('-')[-1])
                        self.current_logger.info(f"\tStep 2/2: Propagating available transforms onto {scan_path}...")
                        self.current_logger.info(f"\t\tAvailable transforms: {available_txs}")
                        self._propagate_register(
                            moving_im=step1,
                            tx_list=available_txs,
                            output_path=cur_output_path
                        )
                        self.current_logger.info(f"\tRegistration successfully completed.")
                    elif self.scan_available[self.intra_subject_template]:
                        # 1/2: register scan_path to the intra_subject_template directly
                        self.current_logger.info(f"\tStep 1/2: Registering {scan_path} to the intra subject template {self.intra_subject_template}...")
                        intra_subject_template_path = self._get_scan_path(self.intra_subject_template)
                        intra_subject_template_path_full = f'{intermediary_origin_dir}/{self.subject}/{self.session}/{intra_subject_template_path}/{self.session}_{intra_subject_template_path}.nii.gz'
                        step1 = self._direct_register(
                            fixed_path=intra_subject_template_path_full,
                            moving_path=scan_path_full,
                            output_path=f'{cur_output_dir}/{self.session}_{scan_path}_direct_reg_to_{self.intra_subject_template}_.nii.gz'
                        )
                        # 2/2: propagate the intra_subject_template's registration to the MNI template onto scan_path
                        available_txs = self._unravel_available_txs(self.intra_subject_template)
                        self.current_logger.info(f"\tStep 2/2: Propagating intra subject template {self.intra_subject_template} registration to MNI template onto {scan_path}...")
                        self.current_logger.info(f"\t\tAvailable transforms: {available_txs}")
                        self._propagate_register(
                            moving_im=step1,
                            tx_list=available_txs,
                            output_path=cur_output_path
                        )
                        self.current_logger.info(f"\tRegistration successfully completed.")
                    else:
                        # 1/1: register scan_path to the MNI template directly
                        self.current_logger.info(f"\tStep 1/1: Neither the intra subject template {self.intra_subject_template} nor an appropriate intermediary scan is available, therefore registering {scan_path} directly to MNI template...")
                        self._direct_register(
                            fixed_path=self.mni_template_path, 
                            moving_path=scan_path_full, 
                            output_path=cur_output_path
                        )
                        self.current_logger.info(f"\tRegistration successfully completed.")
            else:
                # In this case, we want to:
                # 1) register scan_path to the intra_subject_template directly, then;
                # 2) propagate the intra_subject_template's registration to the MNI template onto scan_path.
                # Should we be missing the intra_subject_template, we simply register directly to the MNI template.
                if self.scan_available[self.intra_subject_template]:
                    # 1/2: register scan_path to the intra_subject_template directly
                    self.current_logger.info(f"\tStep 1/2: Registering {scan_path} to the intra subject template {self.intra_subject_template}...")
                    intra_subject_template_path = self._get_scan_path(self.intra_subject_template)
                    intra_subject_template_path_full = f'{self.data_dir}/{self.subject}/{self.session}/{intra_subject_template_path}/{self.session}_{intra_subject_template_path}.nii.gz'
                    step1 = self._direct_register(
                        fixed_path=intra_subject_template_path_full,
                        moving_path=scan_path_full,
                        output_path=f'{cur_output_dir}/{self.session}_{scan_path}_direct_reg_to_{self.intra_subject_template}_.nii.gz'
                    )
                    # 2/2: propagate the intra_subject_template's registration to the MNI template onto scan_path
                    available_txs = self._unravel_available_txs(self.intra_subject_template)
                    self.current_logger.info(f"\tStep 2/2: Propagating intra subject template {self.intra_subject_template} registration to MNI template onto {scan_path}...")
                    self.current_logger.info(f"\t\tAvailable transforms: {available_txs}")
                    self._propagate_register(
                        moving_im=step1, 
                        tx_list=available_txs,
                        output_path=cur_output_path
                    )
                    self.current_logger.info(f"\tRegistration successfully completed.")
                else:
                    # 1/1: register scan_path to the MNI template directly
                    self.current_logger.info(f"\tStep 1/1: Intra subject template {self.intra_subject_template} not available, therefore registering {scan_path} directly to MNI template...")
                    self._direct_register(
                        fixed_path=self.mni_template_path, 
                        moving_path=scan_path_full, 
                        output_path=cur_output_path
                    )
                    self.current_logger.info(f"\tRegistration successfully completed.")

        except Exception as e:
            self.current_logger.info(f"\tError in registration: {e}")
            self.num_failed_registrations += 1
            self.failed_registrations.append(f"{self.session}/{scan_path}")

        # When finished, reset the current logger
        end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.current_logger.info(f"\tRegistration completed at: {end_date}")
        hours, rem = divmod(time.time() - start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        time_elapsed = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
        self.current_logger.info(f"\tTotal time elapsed: {time_elapsed}")
        self.current_handler.close()
        self.current_logger.removeHandler(self.current_handler) 
        self.current_logger = None
        self.current_handler = None
        self.current_formatter = None
    
    def _register_session(self, session):
        self.session = session
        
        # Checks to make sure this method was called correctly
        assert self.subject is not None, "The Registration object must have a subject set."
        assert self.session is not None, "The Registration object must have a session set."
        assert self.session in lsdir(f'{self.data_dir}/{self.subject}'), f"Session {self.session} not found for subject {self.subject}."

        # Identify the scans available needing registration
        self.scan_paths = lsdir(f'{self.data_dir}/{self.subject}/{self.session}')
        scan_types = [scan_path.split('-')[-1] for scan_path in self.scan_paths]
        for scan_type in scan_types: self.scan_available[scan_type] = True
        
        # Register each scan to the MNI template
        for scan_type in self.scan_available.keys():
            if self.scan_available[scan_type]:
                self._register_scan(scan_type)

        # Reset the session information once we are finished
        self._reset_session_info()

    def _register_subject(self, subject):
        """
        Register all the scans in all the sessions for a given subject.
        """
        # TODO: Implement propagating existing tx files for those subjects from the previous round of preprocessing
        if subject in lsdir(self.prev_round_dir):
            print(f"Subject {subject} has already been registered. Skipping...")
            return
                
        # Set the subject number and reset the session and scan availability data
        self.subject = subject
        self._reset_session_info()

        # Loop through each session for the subject, and register all scans in the session
        for session in lsdir(f'{self.data_dir}/{self.subject}'):
            self._register_session(session)
        
        self.subject = None
    
    def register_all(self):
        subjects = lsdir(self.data_dir)
        for s in tqdm(subjects, desc='Registering subjects', total=len(subjects), dynamic_ncols=True, smoothing=0.5):
            self._register_subject(s)
        self._end_logging()

if __name__ == '__main__':
    setup()
    reg = Registration()
    reg.register_all()