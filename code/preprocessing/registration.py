import os
import ants
from utils import lsdir
import logging
import shutil
import time
from datetime import datetime

class Registration:
    def __init__(self):
        """
        Registration class to facilitate registration of scans to MNI template.
        """

        # Global directories
        self.tx_dir = f'data/preprocessing/output/6b_REGISTERED'
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
        self.mni_template = ants.image_read(self.mni_template_path, reorient='IAL')

        # Defining the kind of registration
        self.tx_type = 'Affine'

        # Defining the scans that need to be registered using an intermediary
        self.scans_needing_intermediary = ['AX_SWI', 'AX_DIFFUSION', 'AX_ADC']

        # Setting up main logging
        self.main_logger = logging.getLogger('main_logger')
        self.main_logger.setLevel(logging.INFO)
        self.main_handler = logging.FileHandler(f'{self.output_dir}/log.txt')
        self.main_handler(logging.INFO)
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
        self.main_logger.info(f"\ttx_dir: {self.tx_dir}")
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
            'SAG_3D_T2': False,
        }
        self.scan_paths = None
    
    def end_logging(self):
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
        try:
            for i, t in enumerate(tx):
                suffix = '.'.join(t.split('.')[1:])
                self.current_logger.info(f"Attempting to save transform {i + 1}/{len(tx)} as {output_path}_transform_tx_{i}.{suffix}...")
                shutil.copy(t, f'{output_path}_transform_tx_{i}.{suffix}')
                self.current_logger.info(f"Success!")
        except Exception as e:
            self.current_logger.info(f"Error in saving transforms: {e}")

    def _register_scan(self, scan_type):
        # Obtain the current scan path to be registered
        scan_path = self._get_scan_path(scan_type)

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
        if scan_type == self.intra_subject_template:
            # In this case, we want to register scan_path to the MNI template directly
            raise NotImplementedError("Direct registration to MNI template not yet implemented.")
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
            raise NotImplementedError("Registration using intermediary not yet implemented.")
        else:
            # In this case, we want to:
            # 1) register scan_path to the intra_subject_template directly, then;
            # 2) propagate the intra_subject_template's registration to the MNI template onto scan_path.
            # Should we be missing the intra_subject_template, we simply register directly to the MNI template.
            raise NotImplementedError("Not yet implemented.")

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

    def register_subject(self, subject):
        """
        Register all the scans in all the sessions for a given subject.
        """
        # Set the subject number and reset the session and scan availability data
        self.subject = subject
        self._reset_session_info()

        # Loop through each session for the subject, and register all scans in the session
        for session in lsdir(f'{self.data_dir}/{self.subject}'):
            self._register_session(session)
        
        self.subject = None