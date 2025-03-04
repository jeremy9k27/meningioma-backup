# %%
import os
if not os.getcwd().endswith('code'): os.chdir('..')
# custom functions
from deeplearning.utils import get_segs, get_mris, get_seg_roi_key
from deeplearning.transforms import CenterOnTumor, CubifyVolume
from preprocessing.utils import lsdir, explore_3D_array_with_mask_contour
from radiomics.utils import plot_data_split
# PyTorch imports
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
# standard libraries
import pandas as pd
import numpy as np
# logging helpers
from tqdm import tqdm
from zlib import adler32
import logging

def adler32_hash(input_string):
    """Given an input string, returns an adler hexcode hash unique to that string."""
    input_bytes = input_string.encode('utf-8')    
    hash_value = adler32(input_bytes)
    return hex(hash_value)

class MeningiomaDataset(Dataset):
    def __init__(
        self, 
        task_name, 
        labels_file='data/labels/MeningiomaBiomarkerData.csv', 
        mri_dir='data/preprocessing/output/7b_COMPLETED_PREPROCESSED', 
        pulse_sequences=['T1_POST', 'FLAIR', 'ADC'], 
        seg_dir='data/all_smooth_segs_02-08-25/', 
        seg_rois=[1, 3, 4, 5, 6, 22], 
        transforms=None,
        output_dir='data/pytorch_datasets'
    ):
        """
        Parameters
        ----------
        - task_name (str): Name of the classification task to be performed. Must be a column in the labels file. Can be one of: ['Chr22q', 'MethylationSubgroup', 'Chr1p']
        - labels_file (str): Path to the CSV file containing the labels for each subject.
        - mri_dir (str): Directory containing the MRI images.
        - pulse_sequences (list): List of pulse sequences to include in the dataset.
        - seg_dir (str): Directory containing the segmentation masks. If None, no segmentation masks will be included in the dataset.
        - seg_rois (list): List of region of interest (roi) labels to extract from the segmentation masks.
        - transforms (torchvision.transforms.Compose): Composed list of custom transforms (designed to work on sample) to apply to the data.
        - output_dir (str): Directory to save the dataset to.

        Sample dict keys
        ----------------
        - 'mris': a dictionary of volumetric images, where keys are the pulse sequence names, and values are PyTorch tensors
        - 'label': an int encoding the biomarker label of the subject
        - 'sub_id': an int encoding the subject ID
        - 'session_type': a str providing the session type, e.g. 'brainlab' or 'presurgical'
        - 'segs': a dictionary of volumetric image masks, where keys are the mask code (e.g. 22 for whole tumor), and values are binary PyTorch tensors
        """
        # store input arguments
        self.task_name = task_name
        self.labels_file = labels_file
        self.mri_dir = mri_dir
        self.pulse_sequences = pulse_sequences
        self.seg_dir = seg_dir
        self.seg_rois = seg_rois
        self.transforms = transforms

        # create output directory using adler32 hash of input arguments
        self.hash = adler32_hash(f"{task_name}{labels_file}{mri_dir}{[ps.lower() for ps in pulse_sequences]}{seg_dir}{seg_rois}{transforms}")
        self.output_dir = f"{output_dir}/{self.hash}"
        self.is_new_ds = not os.path.exists(self.output_dir)
        if self.is_new_ds: 
            os.makedirs(f"{self.output_dir}/items")
            os.makedirs(f"{self.output_dir}/plots")

        # set up logging
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(f'{self.output_dir}/log.txt')
        formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        if self.is_new_ds:
            self.logger.info(f"Creating MeningiomaDataset with hash: {self.hash}")
            self.logger.info(f"\ttask_name: {task_name}")
            self.logger.info(f"\tlabels_file: {labels_file}")
            self.logger.info(f"\tmri_dir: {mri_dir}")
            self.logger.info(f"\tpulse_sequences: {pulse_sequences}")
            self.logger.info(f"\tseg_dir: {seg_dir}")
            self.logger.info(f"\tseg_rois: {seg_rois}")
            self.logger.info(f"\ttransforms: {transforms}")

        # read in segmentation file paths
        if self.seg_dir is not None:
            self.seg_paths = [f for f in os.listdir(self.seg_dir) if f.startswith('Segmentation')]

        # read in labels file & extract info
        labels_df = pd.read_csv(self.labels_file)
        labels_df.index = labels_df['Subject Number']
        assert self.task_name in labels_df.columns, f"Task name {self.task_name} not found in labels file {self.labels_file}"
        labels_df = labels_df[labels_df[self.task_name].notna()]
        self.labels = labels_df[self.task_name].astype(int)
        self.num_classes = self.labels.nunique()
        self.labels_key = {0: 'Intact', 1: 'Lost'}
        if self.task_name == 'MethylationSubgroup':
            self.labels_key = {0: 'MI', 1: 'IE', 2: 'HM'}
        
        # get list of subjects
        self.subjects_with_mris = [int(s) for s in lsdir(self.mri_dir)]
        self.subjects_with_labels = self.labels.index
        if self.seg_dir is not None:
            self.subjects_with_segs = [int(s.split(' ')[1].split('.')[0]) for s in self.seg_paths]
            self.subjects = sorted(list(set(self.subjects_with_mris) & set(self.subjects_with_segs) & set(self.subjects_with_labels)))
        else:
            self.subjects = sorted(list(set(self.subjects_with_mris) & set(self.subjects_with_labels)))

        # see subject availability by session and pulse sequence type
        self.subjects_by_session = {'brainlab': [], 'presurgical': [], 'other': []}
        self.subjects_by_pulse_sequence = {}
        for ps in self.pulse_sequences:
            self.subjects_by_pulse_sequence[ps] = []
        
        for s in self.subjects:
            session = lsdir(f'{self.mri_dir}/{s}')[0]
            if 'brainlab' in session.lower():
                self.subjects_by_session['brainlab'].append(s)
            elif 'presurgical' in session.lower():
                self.subjects_by_session['presurgical'].append(s)
            else:
                self.subjects_by_session['other'].append(s)
            
            for scan in lsdir(f'{self.mri_dir}/{s}/{session}'):
                for ps in self.pulse_sequences:
                    if scan.lower().endswith(ps.lower()):
                        self.subjects_by_pulse_sequence[ps].append(s)
                        break
        
        # narrow list of subjects to those with all pulse sequences available
        subjects_set = set(self.subjects)
        for ps in self.pulse_sequences:
            subjects_set &= set(self.subjects_by_pulse_sequence[ps])
        self.subjects = sorted(list(subjects_set))
        self.labels = self.labels[self.subjects]
        for session in self.subjects_by_session:
            self.subjects_by_session[session] = set(self.subjects_by_session[session])
            self.subjects_by_session[session] &= subjects_set
            self.subjects_by_session[session] = sorted(list(self.subjects_by_session[session]))
        
        # get lists of subjects from each class
        self.subjects_by_class = {}
        for k in self.labels_key.keys():
            self.subjects_by_class[k] = self.labels.index[np.where(self.labels == k)[0]].values.tolist()

        # logging
        if self.is_new_ds:
            self.logger.info(f"Number of subjects retrieved using above params: {len(self.subjects)}")
            self.logger.info(f"Subjects list: {self.subjects}")
            self.logger.info('-'*80)
        
        if not self.is_new_ds:
            self.logger.info(f"Loading existing dataset from {self.output_dir}")
            self.logger.info('-'*80)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        # get subject ID
        sub_id = self.subjects[idx]
        if os.path.exists(f"{self.output_dir}/items/{sub_id}.pt"):
            sample = torch.load(f"{self.output_dir}/items/{sub_id}.pt", weights_only=False)
            return sample
        else:
            # get session info
            session = lsdir(f'{self.mri_dir}/{sub_id}')[0]
            session_type = session.split('_')[-1].lower()

            # get mris
            mris = get_mris(subject_mri_dir=f'{self.mri_dir}/{sub_id}/{session}', pulse_sequences=self.pulse_sequences)
            for k in mris.keys(): mris[k] = torch.from_numpy(mris[k])
                        
            # get label
            label = self.labels[sub_id].astype(int)

            # combine all info
            sample = {'mris': mris, 'label': label, 'sub_id': sub_id, 'session_type': session_type}

            # get segmentations if desired
            if self.seg_dir is not None:
                segs = get_segs(subject=sub_id, seg_dir=self.seg_dir, seg_paths=self.seg_paths, rois=self.seg_rois)
                for k in segs.keys(): segs[k] = torch.from_numpy(segs[k])
                sample['segs'] = segs
            
            # apply transforms if desired
            if self.transforms: sample = self.transforms(sample)

            # save sample to disk
            torch.save(sample, f"{self.output_dir}/items/{sub_id}.pt")

            return sample
    
    def precache(self):
        """Iterates through the dataset, saving each sample as a .pt object for fast access later"""
        for sample in tqdm(self, total=len(self), desc="Precaching... "): pass

    def plot_data_split(self):
        """Plots the labels breakdown in 1. the entire dataset; 2. among brainlab subjects; and 3. among presurgical subjects. Saves plots to plot dir."""
        plot_data_split(self.labels[self.subjects].values.astype(int), title=f"All subjects {self.task_name}", output_file=f"{self.output_dir}/plots/data_split_all.png")
        plot_data_split(self.labels[self.subjects_by_session['brainlab']].values.astype(int), title=f"Brainlab subjects {self.task_name}", output_file=f"{self.output_dir}/plots/data_split_brainlab.png")
        plot_data_split(self.labels[self.subjects_by_session['presurgical']].values.astype(int), title=f"Presurgical subjects {self.task_name}", output_file=f"{self.output_dir}/plots/data_split_presurgical.png")
        plot_data_split(self.labels[self.subjects_by_session['other']].values.astype(int), title=f"Other subjects {self.task_name}", output_file=f"{self.output_dir}/plots/data_split_other.png")

    def get_labels(self):
        return self.labels
    
    def get_labels_key(self):
        """Returns a dictionary providing a key to understanding the given task's labels"""
        return self.labels_key
    
    def get_seg_roi_key(self):
        """Returns a dict mapping int keys to corresponding segmentation rois."""
        return get_seg_roi_key()
    
    def get_sample_weights(self):
        """
        Returns an array of sample weights intended to be passed on to torch.utils.data.WeightedRandomSampler,
        where weights are calculated as that sample's inverse class frequency.
        """
        y = self.labels.values.astype(int)
        icf = 1/np.bincount(y)
        return icf[y]

    def get_subjects(self):
        if self.seg_dir is not None:
            return self.subjects, self.subjects_with_mris, self.subjects_with_segs, self.subjects_with_labels
        return self.subjects, self.subjects_with_mris, self.subjects_with_labels

    def get_subjects_by_session(self): return self.subjects_by_session
    def get_subjects_by_class(self): return self.subjects_by_class
    def get_subjects_by_pulse_sequence(self): return self.subjects_by_pulse_sequence

if not os.getcwd().endswith('Meningioma'): os.chdir('..')

# txs = transforms.Compose([
#     CenterOnTumor(cube_size=96, margin=5, pad_size=60)
# ])

# ds = MeningiomaDataset(
#     task_name='MethylationSubgroup',
#     pulse_sequences=['t1_post'],
#     seg_rois=[22],
#     transforms=None
# )

# ds.plot_data_split()

ds2 = MeningiomaDataset(
    task_name='Chr22q',
    pulse_sequences=['t1_post'],
    seg_rois=[22],
    transforms=None
)

# ds.precache()

# %%
sample = ds[0]
cubify = CubifyVolume(cube_size=240)
sample_cube = cubify(sample)

# %%
explore_3D_array_with_mask_contour(sample_cube['mris']['t1_post'].numpy(), sample_cube['segs'][22].numpy())

# %%
for sample in tqdm(ds, total=len(ds)):
    assert 'mris' in sample.keys()
    assert 'label' in sample.keys()
    assert 'sub_id' in sample.keys()
    assert 'session_type' in sample.keys()
    assert 'segs' in sample.keys()

# %%
