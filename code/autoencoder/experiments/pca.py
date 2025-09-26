# %%
import os
import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA
from code.deeplearning.transforms import *
from code.deeplearning.models import *
DEVICE = torch.device(f'cuda:2' if torch.cuda.is_available() else 'cpu')
SEED = 0
torch.manual_seed(SEED)  # Set the seed for CPU random number generators
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)  # Set the seed for GPU random number generators

print("start begin", os.getcwd())
if os.getcwd().endswith('home'): os.chdir('meningioma')
while not os.getcwd().endswith('meningioma'): os.chdir('..')

slice_vectors = []  
source_labels = []         

full_model = CalabreseModelUNetSkip(input_channels=2, layer_layout=[1, 1, 2, 2], original_shape = 96, use_batch=False).to(DEVICE)
full_model.load_state_dict(torch.load('code/deeplearning/weights/unet.pth'))

model = CalabreseModelEncoder(input_channels=2, layer_layout=[1, 1, 2, 2], original_shape = 96, use_batch=False).to(DEVICE)
model.encoder = full_model.encoder

def find_max_cancerous_slice(seg, axis=0):
    """
    Given a volumetric segmentation mask, returns the index of the slice along the provided axis with the maximum tumor content.

    Parameters:
    -----------
    seg (np.ndarray): The volumetric segmentation mask.
    axis (int): The axis along which to find the slice with the maximum tumor content. Default is 0 (axial plane if segmentation loaded in 'IAL' orientation with antspy).

    Returns:
    --------
    (int): The index of the slice with the maximum tumor content along the provided axis.
    """
    axes_to_sum_over = tuple(set(np.arange(seg.ndim)) - {axis})
    return np.argmax(np.sum(seg, axis=axes_to_sum_over))



# %%
if os.getcwd().endswith('home'): os.chdir('meningioma')
while not os.getcwd().endswith('meningioma'): os.chdir('..')
print("start brats", os.getcwd())
from prep_data import *


ds = UnlabeledScansDataset(
    root_dir= 'BraTS-MEN-Train', 
    size = 155,
    transforms=transforms.Compose([
        CenterOnTumor(cube_size=96, margin=5, pad_size=60),
        Normalize2(mean=[0], std=[1])]))


i = 0
with torch.no_grad():
    for idx in range(len(ds.sub_ids)):
        if i % 50 == 0: print(i)
        i += 1
        sample = ds.__getitem__(idx)
        X = torch.stack([sample['mris']['t1c'], sample['mris']['t2f']], dim=0).unsqueeze(0).to(DEVICE)
        latent_vector = model.forward_encoder(X)

        #seg = np.array(sample['segs'][22])
        #t1c = sample['mris']['t1c']
        #most_cancerous_index = find_max_cancerous_slice(seg, axis=0)
        #t1c_slice = t1c[most_cancerous_index, :, :]  # shape: (H, W)
        # Flatten 2D slice into a vector
        
        slice_vectors.append(latent_vector.flatten().cpu().numpy())  # Move to CPU
        source_labels.append('BraTS')




# %%
print(os.getcwd())
if os.getcwd().endswith('home'): os.chdir('yes9029')
if os.getcwd().endswith('yes9029'): os.chdir('meningioma')
if os.getcwd().endswith('code'): os.chdir('..')
#if os.getcwd().endswith('meningioma'): os.chdir('code')
from code.deeplearning.prep_data import MeningiomaDataset
print("start ours", os.getcwd())

ds = MeningiomaDataset(
    task_name='Chr1p',
    pulse_sequences=['t1_post', 'flair'],
    seg_rois=[22],
    transforms=transforms.Compose([
        CenterOnTumor(cube_size=96, margin=5, pad_size=60),
        Normalize2(mean=[0], std=[1])]))

target_labels = []
i = 0
with torch.no_grad():
    for idx in range(len(ds.subjects)):
        if i % 15 == 0: print(i)
        i += 1
        sample = ds.__getitem__(idx)
        X = torch.stack([sample['mris']['t1_post'], sample['mris']['flair']], dim=0).unsqueeze(0).to(DEVICE)
        latent_vector = model.forward_encoder(X)

        #seg = np.array(sample['segs'][22])
        #t1c = sample['mris']['t1_post']
        #most_cancerous_index = find_max_cancerous_slice(seg, axis=0)
        #t1c_slice = t1c[most_cancerous_index, :, :]  # shape: (H, W)
        # Flatten 2D slice into a vector

        slice_vectors.append(latent_vector.flatten().cpu().numpy())  # Move to CPU
        source_labels.append('in house')
        target_labels.append(sample['label'])




# %%
X = np.stack(slice_vectors)  # shape: (num_samples, H*W)

# Fit PCA (choose number of components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Explained variance ratio:", pca.explained_variance_ratio_)


import matplotlib.pyplot as plt
colors = ['red' if source == 'BraTS' else 'blue' for source in source_labels]
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of dense vectors")
# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='red', label='BraTS'),
                   Patch(facecolor='blue', label='in house')]
plt.legend(handles=legend_elements)
plt.show()


# %%
X = np.stack(slice_vectors)  # shape: (num_samples, H*W)

# Fit PCA (choose number of components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Explained variance ratio:", pca.explained_variance_ratio_)  # Fixed typo: added underscore

import matplotlib.pyplot as plt
color_map = {0: 'blue', 1: 'red', 2: 'green'}
colors = [color_map[label] for label in target_labels]
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of dense vectors")

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='blue', label='Class 0'),
                   Patch(facecolor='red', label='Class 1'),
                   Patch(facecolor='green', label='Class 2')]
plt.legend(handles=legend_elements)
plt.show()

# %%