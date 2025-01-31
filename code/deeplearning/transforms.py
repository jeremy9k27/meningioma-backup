import torch
from torch.nn.functional import pad, interpolate

class CenterOnTumor(object):
    def __init__(self, cube_size=96, margin=5, pad_size=60):
        self.cube_size = cube_size
        self.margin = margin
        self.pad_size = pad_size
    
    def __call__(self, sample):
        mris = sample['mris']
        segs = sample['segs']
        
        # pad all volumes with 60 voxels on each side to ensure cropping to a cube centered on tumor is possible
        for k in mris.keys():
            mris[k] = pad(mris[k], pad=(self.pad_size,)*6)
        for k in segs.keys():
            segs[k] = pad(segs[k], pad=(self.pad_size,)*6)
        
        # get whole tumor mask
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