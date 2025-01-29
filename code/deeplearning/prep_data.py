# %%
import os
if not os.getcwd().endswith('code'): os.chdir('..')
from preprocessing.utils import lsdir
from radiomics.utils import plot_data_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# %%
class MeningiomaDataset(Dataset):
    def __init__(self, task_name, mri_dir, seg_dir, labels_file, pulse_sequences=['ADC', 'FLAIR', 'T1_POST', 'DIFFUSION'], transform=None):
        self.task_name = task_name
        self.mri_dir = mri_dir
        self.seg_dir = seg_dir
        self.labels_file = labels_file
        self.pulse_sequences = pulse_sequences
        self.transform = transform

        # read in segmentation file paths
        self.seg_paths = [f for f in os.listdir(self.seg_dir) if f.startswith('Segmentation')]

        # read in labels file & extract info
        labels_df = pd.read_csv(self.labels_file)
        labels_df.index = labels_df['Subject Number']
        assert self.task_name in labels_df.columns, f"Task name {self.task_name} not found in labels file {self.labels_file}"
        labels_df = labels_df[labels_df[self.task_name].notna()]
        self.labels = labels_df[self.task_name]
        self.num_classes = self.labels.nunique()
        self.labels_key = {0: 'Intact', 1: 'Lost'}
        if self.task_name == 'MethylationSubgroup':
            self.labels_key = {0: 'MI', 1: 'IE', 2: 'HM'}
        
        # get list of subjects
        self.subjects_with_mris = [int(s) for s in lsdir(self.mri_dir)]
        self.subjects_with_segs = [int(s.split(' ')[1].split('.')[0]) for s in self.seg_paths]
        self.subjects_with_labels = self.labels.index
        self.subjects = sorted(list(set(self.subjects_with_mris) & set(self.subjects_with_segs) & set(self.subjects_with_labels)))

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
        for session in self.subjects_by_session:
            self.subjects_by_session[session] = sorted(list(set(self.subjects_by_session[session]) & subjects_set))

        plot_data_split(self.labels[self.subjects].values.astype(int), title=f"All subjects {self.task_name}")
        plot_data_split(self.labels[self.subjects_by_session['brainlab']].values.astype(int), title=f"Brainlab subjects {self.task_name}")
        plot_data_split(self.labels[self.subjects_by_session['presurgical']].values.astype(int), title=f"Presurgical subjects {self.task_name}")

    def __len__(self):
        # return len(self.data)
        raise NotImplementedError

    def __getitem__(self, idx):
        # return self.data[idx], self.labels[idx]
        raise NotImplementedError
    
    def get_labels(self):
        return self.labels

    def get_subjects(self):
        return self.subjects, self.subjects_with_mris, self.subjects_with_segs, self.subjects_with_labels
    
# %%
if not os.getcwd().endswith('Meningioma'): os.chdir('..')

ds = MeningiomaDataset(
    task_name='Chr22q',
    mri_dir='data/preprocessing/output/7b_COMPLETED_PREPROCESSED',
    seg_dir='data/all_smooth_segs_12-12-24',
    labels_file='data/labels/MeningiomaBiomarkerData.csv',
    pulse_sequences=['t1_post'],
    transform=None
)
# %%
