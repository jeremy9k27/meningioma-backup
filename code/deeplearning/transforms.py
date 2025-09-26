import torch
from torch.nn.functional import pad, interpolate
from torchvision import transforms

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


import torchvision.transforms.functional as F

def rotate_3d(tensor, angle):
    # tensor shape: [channels, depth, height, width]
    # Rotate each depth slice individually
    rotated_slices = []
    for d in range(tensor.shape[1]):  # Loop through depth dimension
        slice_2d = tensor[:, d, :, :]  # [channels, height, width]
        rotated_slice = F.rotate(slice_2d, angle)
        rotated_slices.append(rotated_slice)
    return torch.stack(rotated_slices, dim=1)
    
class DetRotation3D:
    '''
    this gets called in the training loop, not to be used as part of transform pipeline
    '''
    def __init__(self, degrees=15):
        self.degrees = degrees
    
    def __call__(self, X_batch, subject_ids, epoch):

        augmented_batch = []
        for i, subject_id in enumerate(subject_ids):
            
            sample = X_batch[i]  # Shape: [channels, depth, height, width]
            
            # Create deterministic seed
            seed = hash(f"{subject_id}_{epoch}") % (2**32)
            torch.manual_seed(seed)
            
            angle = torch.randint(-self.degrees, self.degrees, (1,)).item()
            rotated_sample = rotate_3d(sample, angle)  # You'll need this function
            augmented_batch.append(rotated_sample)
        
        return torch.stack(augmented_batch)
