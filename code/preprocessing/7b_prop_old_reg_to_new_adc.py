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
        self.data_dir = 'data/round2_preprocessing/output/6_ZSCORE_NORMALIZED'
        self.output_dir = 'data/round2_preprocessing/output/7b_UPDATED_ADC_REGISTERED'
        self.log_dir = f'{self.output_dir}/logfiles'

        # Create directories if they don't exist
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)
        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)

        # Setting up MNI template files
        self.mni_template_path = f'{self.prev_round_dir}/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii'
        self.mni_template_ants = ants.image_read(self.mni_template_path, reorient='IAL')

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
        self.main_logger.info(f"\tdata_dir: {self.data_dir}")
        self.main_logger.info(f"\toutput_dir: {self.output_dir}")
        self.main_logger.info(f"\tlog_dir: {self.log_dir}")
        self.main_logger.info(f"\tmni_template_path: {self.mni_template_path}")
        self.main_logger.info(f"{self.bar}")

        # Keeping track of the failed registrations
        self.num_failed_registrations = 0
        self.failed_registrations = []

        # Subject-session-specific information
        self.subject = None
        self.session = None
        self.scan_available = { # Note, the order here is used to determine the order of looking for the first available transform to start unravelling the registration history
            'AX_DIFFUSION': False,
            'SAG_3D_FLAIR': False,
            'AX_3D_T1_PRE': False,
            'SAG_3D_T2': False,
            'AX_3D_T1_POST': False,            
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
        self.scan_available = { # Note, the order here is used to determine the order of looking for the first available transform to start unravelling the registration history
            'AX_DIFFUSION': False,
            'SAG_3D_FLAIR': False,
            'AX_3D_T1_PRE': False,
            'SAG_3D_T2': False,
            'AX_3D_T1_POST': False,            
        }
        self.scan_paths = None

    def _get_scan_path(self, scan_type):
        assert self.scan_paths is not None, "The scan_paths attribute must be set before calling this method."
        scan_path = [path for path in self.scan_paths if path.endswith(scan_type)]
        assert len(scan_path) == 1, f"Found {len(scan_path)} paths for scan type {scan_type}."
        return scan_path[0]
    
    def _propagate_register(self, moving_im, tx_list, output_path):
        result = ants.apply_transforms(fixed=self.mni_template_ants, moving=moving_im, transformlist=tx_list, verbose=False)
        result.to_file(output_path)
    
    def _unravel_available_txs(self):
        next_scan = 'AX_ADC'
        for scan_type in self.scan_available.keys():
            if self.scan_available[scan_type]:
                next_scan = scan_type
                break
        
        all_tx_files = []
        while next_scan != 'MNI':
            scan_path = self._get_scan_path(next_scan)
            cur_dir = f'{self.prev_round_dir}/{self.subject}/{self.session}/{scan_path}'
            cur_files = os.listdir(cur_dir)
            tx_files = [f"{cur_dir}/{cf}" for cf in cur_files if '_tx_' in cf]
            all_tx_files.extend(tx_files)
            next_scan = tx_files[0].split('/')[-1].split('_to_')[-1].split('_tx_')[0]
            if '_transform' in next_scan: next_scan = next_scan.split('_transform')[0]
        return all_tx_files
    
    def _register_scan(self, scan_type):
        # Obtain the current scan path to be registered
        scan_path = self._get_scan_path(scan_type)
        scan_path_full = f'{self.data_dir}/{self.subject}/{self.session}/{scan_path}/{self.session}_{scan_path}.nii.gz'
        cur_output_dir = f'{self.output_dir}/{self.subject}/{self.session}/{scan_path}'
        if not os.path.exists(cur_output_dir): os.makedirs(cur_output_dir)
        shutil.copy(
            f'{self.prev_round_dir}/{self.subject}/{self.session}/{scan_path}/{self.session}_{scan_path}.json',
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
            # 1*/1: propagate the AX_DIFFUSION scan's entire registration history to the ADC scan
            self.current_logger.info(f"\tStep 1/1: Propagating available registration history to the AX_ADC scan...")
            available_txs = self._unravel_available_txs()
            self.current_logger.info(f"\t\tAvailable transforms: {available_txs}")
            self._propagate_register(
                moving_im=ants.image_read(scan_path_full, reorient='IAL'), 
                tx_list=available_txs,
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

        # Identify the scans available
        self.scan_paths = lsdir(f'{self.prev_round_dir}/{self.subject}/{self.session}')
        scan_types = [scan_path.split('-')[-1] for scan_path in self.scan_paths]
        for scan_type in scan_types: 
            if scan_type in self.scan_available.keys():
                self.scan_available[scan_type] = True
        
        # Register the ADC using the available scans
        self._register_scan('AX_ADC')

        # Reset the session information once we are finished
        self._reset_session_info()

    def _register_subject(self, subject):
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
    
    def register_all(self):
        subjects = lsdir(self.data_dir)
        old_subjects = lsdir(self.prev_round_dir)
        subjects_needing_updating = [s for s in subjects if s in old_subjects]
        for s in tqdm(subjects_needing_updating, desc='Registering subjects', total=len(subjects_needing_updating), dynamic_ncols=True, smoothing=0.5):
            self._register_subject(s)
        self._end_logging()

if __name__ == '__main__':
    setup()
    reg = Registration()
    reg.register_all()