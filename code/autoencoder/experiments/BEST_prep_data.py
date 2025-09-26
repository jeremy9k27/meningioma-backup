import os
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import torch
from torch.utils.data import random_split, DataLoader
import sys

from code.deeplearning.transforms import *
from zlib import adler32
from radiomics import featureextractor
import SimpleITK as sitk
from datetime import datetime

def adler32_hash(input_string):
    """Given an input string, returns an adler hexcode hash unique to that string."""
    input_bytes = input_string.encode('utf-8')    
    hash_value = adler32(input_bytes)
    return hex(hash_value)

def stack_volumes(volumes):
    """
    Given a BATCHED dictionary of volumes, e.g. batched_sample['mris'], iterate thru all available volumes and stack them in PyTorch's channel dimension (1)
    Return the stacked tensor.
    """
    list_of_volumes = []
    for v in volumes:
        list_of_volumes.append(volumes[v])
    return torch.stack(list_of_volumes, 1)

def concat_pyrad_features(pyrads_batch):
    """
    Given a BATCHED dictionary of pyrad features, e.g. batched_sample['pyrads'], iterate thru all pulse sequence types and stack the corresponding pyrad feature arrays
    """
    tensors = [pyrads_batch[ps] for ps in pyrads_batch.keys()]
    return torch.cat(tensors, dim=1)

def get_normalized_pyrad(sub_ids, extractor):
    all_t1c_features = []
    all_t2f_features = []
    i = 0
    for sub_id in sub_ids:
        if i % 200 == 0: print(sub_id)
        i += 1
        prefix = f"BraTS-MEN-Train/BraTS-MEN-0{sub_id}/BraTS-MEN-0{sub_id}-"
        seg_path = f"{prefix}seg.nii.gz"
        t1c_path = f"{prefix}t1c.nii.gz"
        t2f_path = f"{prefix}t2f.nii.gz"

        seg = nib.load(seg_path).get_fdata()
        seg = ((seg == 1) | (seg == 2) | (seg == 3)).astype(np.uint8)
        t1c = nib.load(t1c_path).get_fdata()
        t2f = nib.load(t2f_path).get_fdata()
                    
        seg_sitk = sitk.GetImageFromArray(seg)
        t1c_sitk = sitk.GetImageFromArray(t1c)
        t2f_sitk = sitk.GetImageFromArray(t2f)

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
    
    normalized_t1c_features = [(feat - t1c_mean) / t1c_std for feat in all_t1c_features]
    normalized_t2f_features = [(feat - t2f_mean) / t2f_std for feat in all_t2f_features]
    
    return normalized_t1c_features, normalized_t2f_features

class UnlabeledScansDataset(Dataset):
    def __init__(self, root_dir, size=96, transforms = None, timestamp = None, output_dir = 'data/pytorch_datasets'):
        self.root_dir = root_dir

        # ids are strings, not ints. eg: '1435-001'
        print("init", os.getcwd())
        self.sub_ids = [d[-8:] for d in os.listdir(root_dir)]

        self.cubify = CubifyVolume(cube_size=size)
        self.transforms = transforms

        # create output directory using adler32 hash of input arguments
        # ASSUMES WE ONLY EVER USE T1C AND T2N
        # for reproducability, dont use previosuyl cached results
        if timestamp:
            self.hash = adler32_hash(f"BraTS{transforms}{timestamp}")
        else:
            self.hash = adler32_hash(f"BraTS{transforms}")
        self.output_dir = f"{output_dir}/{self.hash}"
        self.is_new_ds = not os.path.exists(self.output_dir)
        if self.is_new_ds: 
            os.makedirs(f"{self.output_dir}/items")

        self.extractor = featureextractor.RadiomicsFeatureExtractor("code/autoencoder/experiments/pyrad_params.yml")
        self.normalized_t1c_pyrads, self.normalized_t2f_pyrads = get_normalized_pyrad(self.sub_ids, self.extractor)


    def __len__(self):
        return len(self.sub_ids)


    def __getitem__(self, idx):
        sub_id = self.sub_ids[idx]
        cache_path = f"{self.output_dir}/items/{sub_id}.pt"

        if os.path.exists(cache_path):
            sample = torch.load(cache_path, weights_only=False)
            return sample

        else:
            prefix = f"BraTS-MEN-Train/BraTS-MEN-0{sub_id}/BraTS-MEN-0{sub_id}-"
            seg_path = f"{prefix}seg.nii.gz"
            t1c_path = f"{prefix}t1c.nii.gz"
            t2f_path = f"{prefix}t2f.nii.gz"

            seg = nib.load(seg_path).get_fdata()
            seg = ((seg == 1) | (seg == 2) | (seg == 3)).astype(np.uint8)
            t1c = nib.load(t1c_path).get_fdata()
            t2f = nib.load(t2f_path).get_fdata()

            # get in tensor form
            t1c_pyrad = self.normalized_t1c_pyrads[idx]
            t2f_pyrad = self.normalized_t2f_pyrads[idx]

            seg = torch.tensor(seg, dtype=torch.float32)
            t1c = torch.tensor(t1c, dtype=torch.float32)
            t2f = torch.tensor(t2f, dtype=torch.float32)

            sample = {
                'sub_id': sub_id,
                'mris': {
                    't1c': t1c,
                    't2f': t2f
                },
                'segs': {
                    22: seg
                },
                'pyrads' : {
                    't1c' : t1c_pyrad,
                    't2f' : t2f_pyrad
                }
            }

            # Apply cubification and transforms
            sample = self.cubify(sample)
            if self.transforms:
                sample = self.transforms(sample)

            # Save to cache
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(sample, cache_path)

            return sample


def get_loaders(dataset, val_ratio=0.2, bs=4): #new
    total_size = len(dataset)
    val_size = int(val_ratio * total_size)
    train_size = total_size - val_size
    print(f"train size: {train_size}, val size: {val_size}")

    # Perform the split
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

    return train_loader, val_loader