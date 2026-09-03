import os

if os.getcwd().endswith('meningioma'): os.chdir('code')
if not os.getcwd().endswith('code'): os.chdir('..')
print("prep_data:", os.getcwd())
import sys
sys.path.append('.')
# custom functions
from deeplearning.utils import get_segs, get_mris, get_seg_roi_key
from preprocessing.utils import lsdir
from radiomics_folder.utils import plot_data_split
from deeplearning.transforms import CenterOnTumor
# PyTorch imports
from torchvision import transforms
from torch.utils.data import Dataset, Subset, DataLoader, WeightedRandomSampler
import torch
# standard libraries
import pandas as pd
import numpy as np
# logging helpers
from tqdm import tqdm
from zlib import adler32
import logging
from radiomics import featureextractor
import SimpleITK as sitk

def adler32_hash(input_string):
    """Given an input string, returns an adler hexcode hash unique to that string."""
    input_bytes = input_string.encode('utf-8')    
    hash_value = adler32(input_bytes)
    return hex(hash_value)

class MeningiomaDataset(Dataset):
    def __init__(
        self,
        task_name, 
        labels_file='/home/data/lawrence/meningioma_data/labels/MeningiomaBiomarkerData.csv', 
        mri_dir='/home/data/lawrence/meningioma_data/preprocessing/output/7b_COMPLETED_PREPROCESSED', 
        pulse_sequences=['T1_POST', 'FLAIR', 'ADC'], 
        seg_dir='/home/data/lawrence/meningioma_data/all_smooth_segs_02-08-25/', 
        seg_rois=[1, 3, 4, 5, 6, 22], 
        transforms=None,
        output_dir='data/pytorch_datasets'
    ):
        """
        Parametersload
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
        for k in self.labels_key:
            self.subjects_by_class[k] = self.labels.index[np.where(self.labels == k)[0]].values.tolist()

        # logging
        if self.is_new_ds:
            self.logger.info(f"Number of subjects retrieved using above params: {len(self.subjects)}")
            self.logger.info(f"Subjects list: {self.subjects}")
            self.logger.info('-'*80)
        
        if not self.is_new_ds:
            self.logger.info(f"Loading existing dataset from {self.output_dir}")
            self.logger.info('-'*80)

        #self.extractor = featureextractor.RadiomicsFeatureExtractor("code/autoencoder/experiments/pyrad_params.yml")

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        # get subject ID
        sub_id = self.subjects[idx]
        if os.path.exists(f"{self.output_dir}/items/{sub_id}.pt"):
            sample = torch.load(f"{self.output_dir}/items/{sub_id}.pt", weights_only=False)
            #print(sub_id)

            '''
            
            if 'pyrads' not in sample:
                sample['pyrads'] = []
                seg_sitk = sitk.GetImageFromArray(sample['segs'][22])
                for ps in sample['mris']:
                    mri_sitk = sitk.GetImageFromArray(sample['mris'][ps])
                    pyrad_dict = self.extractor.execute(mri_sitk, seg_sitk)
                    pyrad_array = [float(pyrad_dict[k]) for k in pyrad_dict if isinstance(pyrad_dict[k], (int, float, np.number, np.ndarray)) 
                                                                                and not k.startswith('diagnostics_')]
                    sample['pyrads'].append(pyrad_array)
                sample['pyrads'] = torch.tensor(sample['pyrads']).flatten()
                torch.save(sample, f"{self.output_dir}/items/{sub_id}.pt")  
            ''' 
            return sample
            
        else:
            # get session info
            session = lsdir(f'{self.mri_dir}/{sub_id}')[0]
            session_type = session.split('_')[-1].lower()

            # get mris
            mris = get_mris(subject_mri_dir=f'{self.mri_dir}/{sub_id}/{session}', pulse_sequences=self.pulse_sequences)
            for k in mris: mris[k] = torch.from_numpy(mris[k])
                        
            # get label
            label = self.labels[sub_id].astype(int)

            # combine all info
            sample = {'mris': mris, 'label': label, 'sub_id': sub_id, 'session_type': session_type}

            # get segmentations if desired
            if self.seg_dir is not None:
                segs = get_segs(subject=sub_id, seg_dir=self.seg_dir, seg_paths=self.seg_paths, rois=self.seg_rois)
                for k in segs: segs[k] = torch.from_numpy(segs[k])
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

    def get_all_subject_lists(self):
        if self.seg_dir is not None:
            return self.subjects, self.subjects_with_mris, self.subjects_with_segs, self.subjects_with_labels
        return self.subjects, self.subjects_with_mris, self.subjects_with_labels
    
    def get_normalized_pyrad(sub_ids, extractor):
        all_t1_post_features = []
        all_flair_features = []
        
        for i in range(len(self.subjects)):
            sub_id = self.subjects[idx]

            session = lsdir(f'{self.mri_dir}/{sub_id}')[0]
            session_type = session.split('_')[-1].lower()

            # get mris
            mris = get_mris(subject_mri_dir=f'{self.mri_dir}/{sub_id}/{session}', pulse_sequences=self.pulse_sequences)
            for k in mris: mris[k] = torch.from_numpy(mris[k])


            # get segmentations if desired
            if self.seg_dir is not None:
                segs = get_segs(subject=sub_id, seg_dir=self.seg_dir, seg_paths=self.seg_paths, rois=self.seg_rois)
                for k in segs: segs[k] = torch.from_numpy(segs[k])
                sample['segs'] = segs
            
            # apply transforms if desired
            if self.transforms: sample = self.transforms(sample)





            t1c_pyrad_dict = extractor.execute(t1c_sitk, seg_sitk)
            t2f_pyrad_dict = extractor.execute(t2f_sitk, seg_sitk)

            # Include numpy.ndarray in the type check
            numeric_keys = [k for k in t1c_pyrad_dict.keys()
                            if (k in t2f_pyrad_dict and
                                isinstance(t1c_pyrad_dict[k], (int, float, np.number, np.ndarray)) and
                                isinstance(t2f_pyrad_dict[k], (int, float, np.number, np.ndarray)) and
                                not k.startswith('diagnostics_'))]

            numeric_keys = sorted(numeric_keys)

            # Extract values (convert numpy arrays to scalars)
            t1c_values = [float(t1c_pyrad_dict[k]) for k in numeric_keys]
            t2f_values = [float(t2f_pyrad_dict[k]) for k in numeric_keys]

            # get in tensor form
            t1c_pyrad = torch.tensor(t1c_values, dtype=torch.float32)
            t2f_pyrad = torch.tensor(t2f_values, dtype=torch.float32)
            
            all_t1c_features.append(t1c_pyrad)
            all_t2f_features.append(t2f_pyrad)
        
        print(all_t1c_features)
        # Stack all features to compute statistics
        all_t1c_stacked = torch.stack(all_t1c_features) 
        all_t2f_stacked = torch.stack(all_t2f_features) 
        
        t1c_mean = torch.mean(all_t1c_stacked, dim=0)
        t1c_std = torch.std(all_t1c_stacked, dim=0)
        t2f_mean = torch.mean(all_t2f_stacked, dim=0)
        t2f_std = torch.std(all_t2f_stacked, dim=0)
        
        # Add small epsilon to prevent division by zero
        eps = 1e-8
        t1c_std = torch.clamp(t1c_std, min=eps)
        t2f_std = torch.clamp(t2f_std, min=eps)
        
        print("mean", t1c_mean)
        print("std", t1c_std)
        normalized_t1c_features = [(feat - t1c_mean) / t1c_std for feat in all_t1c_features]
        normalized_t2f_features = [(feat - t2f_mean) / t2f_std for feat in all_t2f_features]

        print(normalized_t1c_features)
        
        return normalized_t1c_features, normalized_t2f_features
    
    
    def get_hash(self): return self.hash
    def get_subjects(self): return self.subjects
    def get_subjects_by_session(self): return self.subjects_by_session
    def get_subjects_by_class(self): return self.subjects_by_class
    def get_subjects_by_pulse_sequence(self): return self.subjects_by_pulse_sequence


    
def get_sample_weights(y):
    """
    Calculates the inverse class frequencies of all classes appearing in y, 
    then returns how much weight to put on each sample in y in order to obtain balanced classes when 
    using a WeightedRandomSampler equipped with those weights.
    """
    icf = 1/np.bincount(y.astype(int))
    return icf[y]

def get_proper_indices(full_list, subset_list):
    """
    Because the Meningioma dataset is indexed via something like e.g. range(len(ds)),
    but we have metadata on subjects based on their subject ID number, we have this function to find the appropriate
    indices of subject IDS as they exist in the Meningioma ds.

    Parameters
    ----------
    full_list: is a list of subject IDs as they appear in the full Meningioma ds
    subset_list: is a list of subject IDs we care about, and want the indices where they appear in the full_list.

    Returns
    -------
    A list of length len(subset_list) providing the indices where those subject IDs in subset_list appear in full_list
    """
    proper_idxs = []
    for element in subset_list:
        proper_idxs.append(full_list.index(element))
    return proper_idxs

def create_dataloaders(ds, bs=10, train_prop=0.8, independent_test_set=True, seed=0, replacement = True):
    """
    Given a Meningioma dataset object, this constructs training, validation, and test set dataloaders,
    returning them in a dictionary. 
    """
    np.random.seed(seed)
    torch.manual_seed(seed) 
    subs_by_class = ds.get_subjects_by_class()

    if independent_test_set:
        # Split train&val vs test by session type
        subs_by_sess = ds.get_subjects_by_session()
        train_val_sub_IDs = subs_by_sess['brainlab']
        test_sub_IDs = subs_by_sess['presurgical'] + subs_by_sess['other']
    else:
        # Split train&val vs test stratified by class
        train_val_sub_IDs = []
        test_sub_IDs = []
        for k in subs_by_class:
            valid_sub_IDs = subs_by_class[k]
            np.random.shuffle(valid_sub_IDs)
            divider = int(round(0.8*len(valid_sub_IDs)))
            train_val_sub_IDs.extend(valid_sub_IDs[:divider])
            test_sub_IDs.extend(valid_sub_IDs[divider:-1])
        np.random.shuffle(train_val_sub_IDs)

    # Stratified train vs val split by class
    train_sub_IDs = []
    val_sub_IDs = []
    for k in subs_by_class:
        valid_sub_IDs = sorted(list((set(train_val_sub_IDs) & set(subs_by_class[k]))))
        np.random.shuffle(valid_sub_IDs)
        train_val_divider = int(round(train_prop*len(valid_sub_IDs)))
        train_sub_IDs.extend(valid_sub_IDs[:train_val_divider])
        val_sub_IDs.extend(valid_sub_IDs[train_val_divider:-1])
    
    np.random.shuffle(train_sub_IDs)
    train_labels = ds.get_labels()[train_sub_IDs]
    train_sample_weights = get_sample_weights(train_labels)

    subs = ds.get_subjects()
    train_idxs = get_proper_indices(full_list=subs, subset_list=train_sub_IDs)
    val_idxs = get_proper_indices(full_list=subs, subset_list=val_sub_IDs)
    test_idxs = get_proper_indices(full_list=subs, subset_list=test_sub_IDs)

    idxs_dict = {'train': train_idxs, 'val': val_idxs, 'test': test_idxs}
    dataloaders_dict = {}
    for ds_idxs in idxs_dict:
        subset_ds = Subset(ds, idxs_dict[ds_idxs])
        sampler = WeightedRandomSampler(train_sample_weights, len(train_sample_weights), replacement=replacement) if ds_idxs == 'train' else None
        dataloaders_dict[ds_idxs] = DataLoader(subset_ds, batch_size=bs, sampler=sampler, pin_memory=True)

    return dataloaders_dict

def create_only_train_val_dataloaders(ds, bs=10, train_prop=0.8, independent_test_set=True, seed=0):
    """
    Given a Meningioma dataset object, this constructs training and validation dataloaders,
    returning them in a dictionary. 
    """
    np.random.seed(seed)
    subs_by_class = ds.get_subjects_by_class()

    if independent_test_set:
        # Split train&val vs test by session type
        subs_by_sess = ds.get_subjects_by_session()
        train_val_sub_IDs = subs_by_sess['brainlab']
        test_sub_IDs = subs_by_sess['presurgical'] + subs_by_sess['other']
    else:
        # Split train&val vs test stratified by class
        train_val_sub_IDs = []
        test_sub_IDs = []
        for k in subs_by_class:
            valid_sub_IDs = subs_by_class[k]
            np.random.shuffle(valid_sub_IDs)
            divider = int(round(0.8*len(valid_sub_IDs)))
            train_val_sub_IDs.extend(valid_sub_IDs[:divider])
            test_sub_IDs.extend(valid_sub_IDs[divider:-1])
        np.random.shuffle(train_val_sub_IDs)

    # Stratified train vs val split by class
    train_sub_IDs = []
    val_sub_IDs = []
    for k in subs_by_class:
        valid_sub_IDs = sorted(list((set(train_val_sub_IDs) & set(subs_by_class[k]))))
        np.random.shuffle(valid_sub_IDs)
        train_val_divider = int(round(train_prop*len(valid_sub_IDs)))
        train_sub_IDs.extend(valid_sub_IDs[:train_val_divider])
        val_sub_IDs.extend(valid_sub_IDs[train_val_divider:-1])
    
    np.random.shuffle(train_sub_IDs)
    train_labels = ds.get_labels()[train_sub_IDs]
    train_sample_weights = get_sample_weights(train_labels)

    subs = ds.get_subjects()
    train_idxs = get_proper_indices(full_list=subs, subset_list=train_sub_IDs)
    val_idxs = get_proper_indices(full_list=subs, subset_list=val_sub_IDs)
    test_idxs = get_proper_indices(full_list=subs, subset_list=test_sub_IDs)

    idxs_dict = {'train': train_idxs, 'val': val_idxs + test_idxs}
    dataloaders_dict = {}
    for ds_idxs in idxs_dict:
        subset_ds = Subset(ds, idxs_dict[ds_idxs])
        sampler = WeightedRandomSampler(train_sample_weights, len(train_sample_weights), replacement=True) if ds_idxs == 'train' else None
        dataloaders_dict[ds_idxs] = DataLoader(subset_ds, batch_size=bs, sampler=sampler, pin_memory=True)

    return dataloaders_dict

def create_only_train_val_dataloaders_loocv(ds, bs=10, train_ids=None, val_ids=None):
    """
    Construct training and validation dataloaders given subject ID lists.
    no stratification
    """
    subs = ds.get_subjects()
    train_idxs = get_proper_indices(full_list=subs, subset_list=train_ids)
    val_idxs = get_proper_indices(full_list=subs, subset_list=val_ids)

    train_labels = ds.get_labels()[train_ids]
    train_sample_weights = get_sample_weights(train_labels)

    idxs_dict = {'train': train_idxs, 'val': val_idxs}
    dataloaders_dict = {}
    for split in idxs_dict:
        subset_ds = Subset(ds, idxs_dict[split])
        sampler = WeightedRandomSampler(train_sample_weights, len(train_sample_weights), replacement=True) if split == 'train' else None
        dataloaders_dict[split] = DataLoader(subset_ds, batch_size=bs, sampler=sampler, pin_memory=True)

    return dataloaders_dict

def create_train_minival_val_dataloaders_loocv(ds, bs=10, train_ids=None, mini_val_ids=None, val_ids=None):
    """
    Construct training, mini-validation, and validation dataloaders given subject ID lists.
    Maintains weighted sampling for training set only.
    """
    subs = ds.get_subjects()
    train_idxs = get_proper_indices(full_list=subs, subset_list=train_ids)
    mini_val_idxs = get_proper_indices(full_list=subs, subset_list=mini_val_ids)
    val_idxs = get_proper_indices(full_list=subs, subset_list=val_ids)
    
    # Only calculate sample weights for training set
    train_labels = ds.get_labels()[train_ids]
    train_sample_weights = get_sample_weights(train_labels)
    
    idxs_dict = {'train': train_idxs, 'mini_val': mini_val_idxs, 'val': val_idxs}
    dataloaders_dict = {}
    
    for split in idxs_dict:
        subset_ds = Subset(ds, idxs_dict[split])
        # Only use weighted sampler for training
        sampler = WeightedRandomSampler(train_sample_weights, len(train_sample_weights), replacement=True) if split == 'train' else None
        dataloaders_dict[split] = DataLoader(subset_ds, batch_size=bs, sampler=sampler, pin_memory=True)
    
    return dataloaders_dict
    
def stack_volumes(volumes):
    """
    Given a BATCHED dictionary of volumes, e.g. batched_sample['mris'], iterate thru all available volumes and stack them in PyTorch's channel dimension (1)
    Return the stacked tensor.
    """
    list_of_volumes = []
    for v in volumes:
        list_of_volumes.append(volumes[v])
    return torch.stack(list_of_volumes, 1)