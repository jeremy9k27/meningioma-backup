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

#full_model = CalabreseModelUNetSkip(input_channels=2, layer_layout=[1, 1, 2, 2], original_shape = 96, use_batch=False).to(DEVICE)
#full_model.load_state_dict(torch.load('code/deeplearning/weights/unet.pth'))

model = CalabreseModelEncoder(input_channels=2, layer_layout=[1, 1, 2, 2], original_shape = 96, use_batch=False).to(DEVICE)
#model.encoder = full_model.encoder
model.load_state_dict(torch.load("code/deeplearning/weights/22q/22q_unfreeze_constant_fold1_cycle0.pth"))
model.eval()

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
print(os.getcwd())
if os.getcwd().endswith('home'): os.chdir('yes9029')
if os.getcwd().endswith('yes9029'): os.chdir('meningioma')
if os.getcwd().endswith('code'): os.chdir('..')
#if os.getcwd().endswith('meningioma'): os.chdir('code')
from code.deeplearning.prep_data import MeningiomaDataset
print("start ours", os.getcwd())

ds = MeningiomaDataset(
    task_name='Chr22q',
    pulse_sequences=['t1_post', 'flair'],
    seg_rois=[22],
    transforms=transforms.Compose([
        CenterOnTumor(cube_size=96, margin=5, pad_size=60),
        Normalize2(mean=[0], std=[1])]))




target_labels = []
pred_labels = []
i = 0
with torch.no_grad():
    for idx in range(len(ds.subjects)):
        if i % 15 == 0: print(i)
        i += 1
        sample = ds.__getitem__(idx)

        X = torch.stack([sample['mris']['t1_post'], sample['mris']['flair']], dim=0).unsqueeze(0).to(DEVICE)
        
        latent_vector = model.forward_projector(X)
        
        y_pred = model(X)  # full forward pass

        #if idx in val_idx:
         #   print("pred:", y_pred)

        y_pred = (y_pred > 0.5).int().cpu()
        pred_labels.extend(y_pred.flatten())


        #seg = np.array(sample['segs'][22])
        #t1c = sample['mris']['t1_post']
        #most_cancerous_index = find_max_cancerous_slice(seg, axis=0)
        #t1c_slice = t1c[most_cancerous_index, :, :]  # shape: (H, W)
        # Flatten 2D slice into a vector

        slice_vectors.append(latent_vector.flatten().cpu().numpy())
        #slice_vectors.append(X.flatten().cpu().numpy())  # Move to CPU
        source_labels.append('in house')
        target_labels.append(sample['label'])



# %%
from sklearn.model_selection import StratifiedKFold
from matplotlib.patches import Patch
X = np.stack(slice_vectors)  # shape: (num_samples, H*W)

# Fit PCA (choose number of components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Explained variance ratio:", pca.explained_variance_ratio_)  

subject_ids = np.array(ds.subjects)
all_labels = np.array(target_labels)
n_splits = 5
fold_to_plot = 3 #0 indexed 

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
splits = list(skf.split(subject_ids, all_labels))
train_idx, val_idx = splits[fold_to_plot]
print("val idx and subject ids:", val_idx, subject_ids[val_idx])

y_true = np.array(target_labels)[train_idx]
y_pred = np.array(pred_labels)[train_idx]
# Boolean mask of correct predictions
correct_mask = (y_pred == y_true)

color_map = {0: 'blue', 1: 'red'}
colors = [color_map[label] for label in y_true]

import matplotlib.pyplot as plt
# Create figure
plt.figure(figsize=(8, 6))

# Correct predictions (circles)
plt.scatter(
    X_pca[train_idx][correct_mask, 0],
    X_pca[train_idx][correct_mask, 1],
    c=np.array(colors)[correct_mask],
    alpha=0.9,
    marker='o',
    s=50,
    label='Correct'
)

# Incorrect predictions (x)
plt.scatter(
    X_pca[train_idx][~correct_mask, 0],
    X_pca[train_idx][~correct_mask, 1],
    c=np.array(colors)[~correct_mask],
    alpha=0.9,
    marker='x',
    s=100,
    linewidths=2,
    label='Incorrect'
)

legend_elements = [
    Patch(facecolor='blue', label='True Class 0'),
    Patch(facecolor='red', label='True Class 1')
]
plt.legend(handles=legend_elements + [
    Patch(facecolor='none', edgecolor='black', label='Correct (o)'),
    Patch(facecolor='none', edgecolor='black', label='Incorrect (x)')
])

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Train Samples: PCA with Prediction Accuracy")
plt.show()



#%% val
y_true = np.array(target_labels)[val_idx]
y_pred = np.array(pred_labels)[val_idx]
correct_mask = (y_pred == y_true)

color_map = {0: 'blue', 1: 'red'}
colors = [color_map[label] for label in y_true]


# Create figure
plt.figure(figsize=(8, 6))

# Correct predictions (circles)
plt.scatter(
    X_pca[val_idx][correct_mask, 0],
    X_pca[val_idx][correct_mask, 1],
    c=np.array(colors)[correct_mask],
    alpha=0.9,
    marker='o',
    s=50,
    label='Correct'
)

# Incorrect predictions (x)
plt.scatter(
    X_pca[val_idx][~correct_mask, 0],
    X_pca[val_idx][~correct_mask, 1],
    c=np.array(colors)[~correct_mask],
    alpha=0.9,
    marker='x',
    s=100,
    linewidths=2,
    label='Incorrect'
)

legend_elements = [
    Patch(facecolor='blue', label='True Class 0'),
    Patch(facecolor='red', label='True Class 1')
]
plt.legend(handles=legend_elements + [
    Patch(facecolor='none', edgecolor='black', label='Correct (o)'),
    Patch(facecolor='none', edgecolor='black', label='Incorrect (x)')
])

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Validation Samples: PCA with Prediction Accuracy")
plt.show()




### UNUSED

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


#%%
# color based on source
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