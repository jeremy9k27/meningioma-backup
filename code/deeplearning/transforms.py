import torch
from torch.nn.functional import pad, interpolate, affine_grid, grid_sample
from torchvision import transforms
import math
import nibabel as nib
import numpy as np

class Normalize(object):
    """
    Wrapper for torchvision.transforms.Normalize fn to work with our Meningioma samples
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.tx = transforms.Normalize(mean=self.mean, std=self.std)

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"
    
    def __call__(self, sample):
        mris = sample['mris']
        for k in mris: 
            print("before", mris[k].mean(), mris[k].std())
            mris[k] = self.tx(mris[k])
            print("after", mris[k].mean(), mris[k].std())
        return sample

class Normalize2(object):
    """
    i think normalize was implemented incorrectly and was essentially a no-op
    init and repr dunders seem meaningless now?
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.tx = transforms.Normalize(mean=self.mean, std=self.std)

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"
    
    def __call__(self, sample):
        mris = sample['mris']
        for k in mris:
            
            # Get THIS tensor's actual statistics
            actual_mean = mris[k].mean()
            actual_std = mris[k].std()
            
            # Normalize to mean=0, std=1
            mris[k] = (mris[k] - actual_mean) / actual_std

        return sample

class CubifyVolume(object):
    """
    Resizes all the volumes of a sample to be of shape [cube_size x cube_size x cube_size]
    """
    def __init__(self, cube_size=96):
        self.cube_size = cube_size
    
    def __repr__(self):
        return f"{self.__class__.__name__}(cube_size={self.cube_size})"

    def __call__(self, sample):
        # get mris and (when available) segs dicts from the input sample
        mris = sample['mris']
        if 'segs' in sample.keys():
            segs = sample['segs']
        else:
            segs = None
        
        # find the current (original) shape of the mris and segs (expected to all be the same)
        pulse_seq = list(sample['mris'].keys())[0]
        current_shape = torch.tensor(sample['mris'][pulse_seq].shape)

        # figure out the largest dim, then use that to find which multiple of 16 to scale the whole volume up to 
        max_dim_size = current_shape.max()
        multiples_of_16 = torch.tensor([16*i for i in range(1, 40)])
        pad_up_idx = torch.where(multiples_of_16 - max_dim_size >= 0)[0][0].item()
        pad_up_size = multiples_of_16[pad_up_idx]

        # calculate the how much to pad all sides of the volume to bring it up to pad_up_size x pad_up_size x pad_up_size
        padding1 = (pad_up_size - current_shape) // 2 + current_shape % 2
        padding2 = (pad_up_size - current_shape) // 2
        padding = torch.empty(6)
        for i in range(len(padding1)):
            padding[2*i] = padding1[i]
            padding[2*i + 1] = padding2[i]
        padding = tuple(padding.int().tolist())[::-1]

        # pad all available volumes to yield a cube of shape pad_up_size x pad_up_size x pad_up_size
        for k in mris.keys():
            mris[k] = pad(mris[k], pad=padding)
        if segs:
            for k in segs.keys():
                segs[k] = pad(segs[k], pad=padding)
        
        assert torch.all(torch.tensor(mris[pulse_seq].shape) == pad_up_size), "An error occurred during the padding step. Debug padding?"

        # resize to be of shape self.cube_size x self.cube_size x self.cube_size
        for k in mris.keys():
            mris[k] = interpolate(mris[k].unsqueeze(0).unsqueeze(0), size=(self.cube_size,)*3, mode='trilinear').squeeze()
        if segs:
            for k in segs.keys():
                segs[k] = interpolate(segs[k].unsqueeze(0).unsqueeze(0).float(), size=(self.cube_size,)*3, mode='nearest-exact').squeeze()

        assert torch.all(torch.tensor(mris[pulse_seq].shape) == torch.tensor(self.cube_size)), "An error occurred during the interpolation step. Debug interpolation?"

        return sample

class CenterOnTumor(object):
    """
    Resizes all the volumes of a sample to be a [cube_size x cube_size x cube_size] voxel centered on the tumor, 
    which includes a [margin x margin x margin] around it. The pad_size is used
    to avoid index out of bounds errors for those tumors close to the edge of a volume.
    """
    def __init__(self, cube_size=96, margin=5, pad_size=60):
        self.cube_size = cube_size
        self.margin = margin
        self.pad_size = pad_size
    
    def __repr__(self):
        return f"{self.__class__.__name__}(cube_size={self.cube_size}, margin={self.margin}, pad_size={self.pad_size})"
    
    def __call__(self, sample):
        # get mris and segs dicts from the input sample
        mris = sample['mris']
        segs = sample['segs']
        
        # pad all volumes with 60 voxels on each side to ensure cropping to a cube centered on tumor is possible
        for k in mris.keys():
            mris[k] = pad(mris[k], pad=(self.pad_size,)*6)
        for k in segs.keys():
            segs[k] = pad(segs[k], pad=(self.pad_size,)*6)
        
        # get whole tumor mask
        assert 22 in segs.keys(), "Segmentation key 22 not found in the provided segmentations. 22 = whole tumor mask needed to center on tumor."
        whole_tumor_mask = segs[22]

        # get bounding box of tumor mask
        tumor_voxels = torch.nonzero(whole_tumor_mask)
        min_coords = tumor_voxels.min(dim=0).values
        max_coords = tumor_voxels.max(dim=0).values

        # apply margin
        min_coords = torch.clamp(min_coords - self.margin, min=0)
        max_coords = torch.clamp(max_coords + self.margin, max=torch.tensor(whole_tumor_mask.shape))

        # calculate center of bounding box
        center_coords = (min_coords + max_coords) // 2

        # calculate the bounding box shape
        bbox_shape = max_coords - min_coords

        # use the largest dimension to determine the size of the original cube
        max_dim = bbox_shape.max().item()

        # calculate new min and max coords
        new_max_coords = torch.clamp(center_coords + (max_dim // 2) + (max_dim % 2), max=torch.tensor(whole_tumor_mask.shape))
        new_min_coords = torch.clamp(center_coords - max_dim // 2, min=0)
        
        # sanity check that the new bbox is a cube
        new_bbox_shape = new_max_coords - new_min_coords
        assert (new_bbox_shape == max_dim).all(), "Unable to make bbox into a perfect cube. Try increasing pad_size or decreasing margin."

        # crop all volumes using the new max and min coords
        # then resize all volumes to self.cube_size x self.cube_size x self.cube_size
        for k in mris.keys():
            mris[k] = mris[k][new_min_coords[0]:new_max_coords[0], new_min_coords[1]:new_max_coords[1], new_min_coords[2]:new_max_coords[2]]
            mris[k] = interpolate(mris[k].unsqueeze(0).unsqueeze(0), size=(self.cube_size,)*3, mode='trilinear').squeeze()
        
        for k in segs.keys():
            segs[k] = segs[k][new_min_coords[0]:new_max_coords[0], new_min_coords[1]:new_max_coords[1], new_min_coords[2]:new_max_coords[2]]
            segs[k] = interpolate(segs[k].unsqueeze(0).unsqueeze(0).float(), size=(self.cube_size,)*3, mode='nearest-exact').squeeze()

        sample['mris'] = mris
        sample['segs'] = segs

        return sample

class BatchedCenterOnTumor:
    def __init__(self, center_on_tumor):
        self.center_on_tumor = center_on_tumor

    def __call__(self, batched_sample):
        
        augmented_samples = []
        batch_size = list(batched_sample["mris"].values())[0].shape[0]
        
        # Process each sample in the batched sample
        for i in range(batch_size):
            single_sample = {
                "mris": {modality: mri_tensor[i] for modality, mri_tensor in batched_sample["mris"].items()},
                "segs": {seg_key: seg_tensor[i] for seg_key, seg_tensor in batched_sample["segs"].items()}
            }
            
            # Apply center_on_tumor to single sample
            augmented_sample = self.center_on_tumor(single_sample)            
            augmented_samples.append(augmented_sample)
        
        # Rebatch: stack all samples back together
        rebatched = {
            "mris": {
                modality: torch.stack([sample["mris"][modality] for sample in augmented_samples], dim=0)
                for modality in augmented_samples[0]["mris"].keys()
            },
            "segs": {
                seg_key: torch.stack([sample["segs"][seg_key] for sample in augmented_samples], dim=0)
                for seg_key in augmented_samples[0]["segs"].keys()
            },
            "label" : batched_sample['label'],
            "sub_id" : batched_sample['sub_id']
        }
        return rebatched


def rotate_3d(volume, mode, angle_x=0, angle_y=0, angle_z=0):

    device = volume.device
    dtype = volume.dtype

    # Start with identity matrix
    R = torch.eye(3, device=device, dtype=dtype)

    batch_size = volume.shape[0]

    # Apply X rotation
    if angle_x != 0:
        angle = math.radians(angle_x)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        R_x = torch.tensor([
            [1,     0,      0],
            [0, cos_a, -sin_a],
            [0, sin_a,  cos_a]
        ], device=device, dtype=dtype)
        R = R @ R_x

    # Apply Y rotation
    if angle_y != 0:
        angle = math.radians(angle_y)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        R_y = torch.tensor([
            [ cos_a, 0, sin_a],
            [ 0,     1,    0 ],
            [-sin_a, 0, cos_a]
        ], device=device, dtype=dtype)
        R = R @ R_y

    # Apply Z rotation
    if angle_z != 0:
        angle = math.radians(angle_z)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        R_z = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,         0,  1]
        ], device=device, dtype=dtype)
        R = R @ R_z

    # Build affine matrix
    affine = torch.zeros((batch_size, 3, 4), device=device, dtype=dtype)
    affine[:, :, :3] = R

    grid = torch.nn.functional.affine_grid(affine, size=volume.shape, align_corners=False)
    rotated = torch.nn.functional.grid_sample(volume, grid, mode=mode, padding_mode="border", align_corners=False)
    return rotated


class DetRotation3D:
    '''
    this gets called in the training loop, not to be used as part of transform pipeline
    to be called on a batch
    '''
    def __init__(self):
        pass
    
    def __call__(self, X_batch, subject_ids, epoch):
        
        augmented_batch = []
        for i, subject_id in enumerate(subject_ids):
            
            sample = X_batch[i]  # Shape: [channels, depth, height, width]

            # Get original cube size
            original_size = sample.shape[1]  # assuming cube: depth=height=width
            
            # Calculate padding needed for 45 degree rotation
            new_size = math.ceil(original_size * math.sqrt(2))
            pad_total = new_size - original_size
            pad_per_side = pad_total // 2
            pad_remainder = pad_total % 2
            
            # Get min value for this sample
            min_val = sample.min()
            
            # Pad equally on all sides (depth, height, width)
            # pad format: (left, right, top, bottom, front, back)
            padded_sample = torch.nn.functional.pad(
                sample, 
                (pad_per_side, pad_per_side + pad_remainder,  # width
                 pad_per_side, pad_per_side + pad_remainder,  # height
                 pad_per_side, pad_per_side + pad_remainder), # depth
                value=min_val
            )
            
            # Randomly choose plane (0=XY/Z-axis, 1=XZ/Y-axis, 2=YZ/X-axis) and degree
            axis = ["z", "y", "x"][torch.randint(0, 3, (1,)).item()]
            angle_deg = [-45,0,45][torch.randint(0, 3, (1,)).item()]
            
            # Apply rotation
            if angle_deg != 0:
                rotated_sample = rotate_3d(padded_sample, angle_deg=angle_deg, axis=axis)
                tensor_np = rotated_sample.squeeze().cpu().numpy()[0]
                print(tensor_np.shape)

                # Save as NIfTI
                nifti_img = nib.Nifti1Image(tensor_np, affine=np.eye(4))
                nib.save(nifti_img, f"rotated{angle_deg}{axis}.nii.gz")
                print("hi")
            augmented_batch.append(rotated_sample)

        return torch.cat(augmented_batch, dim=0)

class CalabreseRotation3D:
    '''
    random rotation in each axis from 0-90
    to be called on a dictionary-like sample from __getitem__
    '''
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
    
    def __call__(self, sample, angle_x=None, angle_y=None, angle_z=None):
        
        # Generate rotation angles
        angles = torch.tensor([0, 45, 90])

        if angle_x is None:
            angle_x = angles[torch.randint(0, 3, (1,))].item()

        if angle_y is None:
            angle_y = angles[torch.randint(0, 3, (1,))].item()

        if angle_z is None:
            angle_z = angles[torch.randint(0, 3, (1,))].item()
        
        # Rotate MRIs with bilinear
        mri_list = [mri_tensor.unsqueeze(1) for mri_tensor in sample["mris"].values()]
        mris_combined = torch.cat(mri_list, dim=1)
        mris_on_device = mris_combined.to(self.device)
        rotated_mris = rotate_3d(mris_on_device, angle_x=angle_x, angle_y=angle_y, angle_z=angle_z, mode='bilinear')
        rotated_mris_cpu = rotated_mris.cpu()
        
        # Rotate segmentations with nearest
        seg_combined = sample["segs"][22].unsqueeze(1).float()
        seg_on_device = seg_combined.to(self.device)
        rotated_seg = rotate_3d(seg_on_device, angle_x=angle_x, angle_y=angle_y, angle_z=angle_z, mode='nearest')
        rotated_seg_cpu = rotated_seg.cpu()
        
        # Build output
        augmented_sample = {"mris": {}, "segs": {}}

        modality_names = list(sample["mris"].keys())
        for i, modality_name in enumerate(modality_names):
            augmented_sample["mris"][modality_name] = rotated_mris_cpu[:, i]

        augmented_sample["segs"][22] = rotated_seg_cpu[:, 0]
        augmented_sample['label'] = sample['label']
        augmented_sample['sub_id'] = sample['sub_id']
    
        
        return augmented_sample


class WorstCasePad:
    '''
    called on individual samples
    '''
    def __call__(self, sample):
        for k in sample["mris"]:
            sample["mris"][k] = self.pad_volume(sample["mris"][k])

        sample["segs"][22] = self.pad_volume(sample["segs"][22])
        return sample

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def pad_volume(self, volume):
        original_size = volume.shape[1]

        new_size = math.ceil(original_size * math.sqrt(2))
        pad_total = new_size - original_size
        pad_per_side = pad_total // 2
        pad_remainder = pad_total % 2

        min_val = volume.min()

        return torch.nn.functional.pad(
            volume,
            (pad_per_side, pad_per_side + pad_remainder,
             pad_per_side, pad_per_side + pad_remainder,
             pad_per_side, pad_per_side + pad_remainder),
            value=min_val
        )
