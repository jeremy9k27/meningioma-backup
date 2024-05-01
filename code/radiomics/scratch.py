# # %%
# import collageradiomics
# import SimpleITK as sitk
# import numpy as np
# import os
# import sys
# import pandas as pd
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)

# from preprocessing.utils import setup

# setup()

# mri_dir = 'data/preprocessed_mri_scans/7_COMPLETED_PREPROCESSED'
# seg_dir = 'data/segmentations'
# im1 = sitk.ReadImage(f'{mri_dir}/6/6_Brainlab/12-AX_3D_T1_POST/6_Brainlab_12-AX_3D_T1_POST.nii.gz')
# mask1 = sitk.ReadImage(f'{seg_dir}/Segmentation 6.nii')

# textures = collageradiomics.Collage(sitk.GetArrayFromImage(im1), sitk.GetArrayFromImage(mask1)).execute()

# %%
import collageradiomics
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup

setup()

# %%
# Load user data in this cell

# Jupyter user: set this to define whether to test 3D calculations or not
use_3D = True

# Read sample image (reads the grey image as an RGB array, so a 3D numpy array)
mri_dir = 'data/preprocessed_mri_scans/7_COMPLETED_PREPROCESSED'
seg_dir = 'data/segmentations'
# image_sitk = sitk.ReadImage(f'{mri_dir}/6/6_Brainlab/12-AX_3D_T1_POST/6_Brainlab_12-AX_3D_T1_POST.nii.gz')
# mask_image = sitk.ReadImage(f'{seg_dir}/Segmentation 6.nii')
image_sitk = sitk.ReadImage('data/samples/BrainSliceTumor.png')
mask_image = sitk.ReadImage('data/samples/BrainSliceTumorMask.png')
image_array = sitk.GetArrayFromImage(image_sitk)
mask_array = sitk.GetArrayFromImage(mask_image)

# %%
if use_3D:
    # flip the first and last slice so there's some gradient in the Z dimension
    image_array[:,:,0] = np.flip(image_array[:,:,0],0)
    image_array[:,:,2] = np.flip(image_array[:,:,1],1)
else:
    # extract a single slice
    image_array = image_array[:,:,0]
    mask_array  = mask_array [:,:,0]

# %%
# Helper functions for visualization

def show_colored_image(figure, axis, image_data, colormap=plt.cm.jet):
    """Helper method to show a colored image in matplotlib.


        :param figure: figure upon which to display
        :type figure: matplotlib.figure.Figure
        :param axis: axis upon which to display
        :type axis: matplotlib.axes.Axes
        :param image_data: image to display
        :type image_data: numpy.ndarray
        :param colormap: color map to convert for display. Defaults to plt.cm.jet.
        :type colormap: matplotlib.colors.Colormap, optional
    """

    if image_data.ndim == 3:
        image_data = image_data[:,:,94]
    image = axis.imshow(image_data, cmap=colormap)
    divider = make_axes_locatable(axis)
    colorbar_axis = divider.append_axes("right", size="5%", pad=0.05)
    figure.colorbar(image, cax=colorbar_axis)


def create_highlighted_rectangle(x, y, w, h):
    """Creates a matplotlib Rectangle object for a highlight effect


        :param x: x location to start rectangle
        :type x: int
        :param y: y location to start rectangle
        :type y: int
        :param w: width of rectangle
        :type w: int
        :param h: height of rectangle
        :type h: int

        :returns: Rectangle used to highlight within a plot
        :rtype: matplotlib.patches.Rectangle
    """
    return Rectangle((x, y), w, h, linewidth=3, edgecolor='cyan', facecolor='none')


def highlight_rectangle_on_image(image_data, min_x, min_y, w, h, colormap=plt.cm.gray):
    """Highlights a rectangle on an image at the passed in coordinate.


        :param image_data: image to highlight
        :type image_data: numpy.ndarray
        :param min_x: x location to start highlight
        :type min_x: int
        :param min_y: y location to start highlight
        :type min_y: int
        :param w: width of highlight rectangle
        :type w: int
        :param h: height of highlight rectangle
        :type h: int
        :param colormap: color map to convert for display. Defaults to plt.cm.jet.
        :type colormap: matplotlib.colors.Colormap, optional

        :returns: image array with highlighted rectangle
        :rtype: numpy.ndarray
    """
    figure, axes = plt.subplots(1, 2, figsize=(15, 15))

    # Highlight window within image.
    show_colored_image(figure, axes[0], image_data, colormap)
    axes[0].add_patch(create_highlighted_rectangle(min_x, min_y, w, h))

    # Crop window.
    cropped_array = image_data[min_y:min_y + h, min_x:min_x + w]
    axes[1].set_title(f'Cropped Region ({w}x{h})')
    show_colored_image(figure, axes[1], cropped_array, colormap)

    plt.show()

    return cropped_array

# %%
# Show slice with mask
figure = plt.figure(figsize = (10, 10))

extent = 0, image_array.shape[1], 0, image_array.shape[0]

# show the image
plt.imshow(image_array[:,:,94] if use_3D else image_array, cmap = plt.cm.gray, extent=extent)

# overlay the mask
plt.imshow(mask_array[:,:,94] if use_3D else mask_array, cmap = plt.cm.jet, alpha=0.3, extent=extent)

plt.title('Input image')

figure.axes[0].get_xaxis().set_visible(False)
figure.axes[0].get_yaxis().set_visible(False)

print(image_array.shape)

# %%
# Example of opti__init__.pyonal parameters
collage = collageradiomics.Collage(
    image_array, 
    mask_array, 
    svd_radius=5, 
    verbose_logging=False,
    num_unique_angles=64
)

# %%
# Run CoLlage Algorithm.Prepare
full_images = collage.execute()

# %%
# Display gradient
figure, axes = plt.subplots(1, 3, figsize=(15, 15))
show_colored_image(figure, axes[0], collage.dx)
axes[0].set_title(f'Gx size={collage.dx.shape}')
show_colored_image(figure, axes[1], collage.dy)
axes[1].set_title(f'Gy size={collage.dy.shape}')
show_colored_image(figure, axes[2], collage.dz)
axes[2].set_title(f'Gz size={collage.dz.shape}')

# %%
# Display dominant angles
figure, axes = plt.subplots(1, 2, figsize=(15, 15))
print(collage.dominant_angles.shape)
show_colored_image(figure, axes[0], collage.dominant_angles[:,:,:,0])
if use_3D:
    show_colored_image(figure, axes[1], collage.dominant_angles[:,:,:,1])
    axes[1].set_title('Secondary Angles: arctan(dz/(dx^2+dy^2))')
else:
    axes[1].set_title('(Unused in 2D mode)')
axes[0].set_title('Dominant Angles: arctan(dy/dx)')

# %%
# Display haralick
figure, axes = plt.subplots(3, 5, figsize=(15,15))

for row in range(3):
    for col in range(5):
        feature = row*5+col
        axis = axes[row][col]
        axis.set_axis_off()
        if feature>=13:
            continue
        collage_output = collage.get_single_feature_output(feature)
        if use_3D:
            collage_output = collage_output[:,:,:,0] # use dominant angle
        show_colored_image(figure, axis, collage_output)
        axis.set_title(f'Collage {feature+1}')

# %%
# Extract a single collage feature by name and overlay it
which_feature = collageradiomics.HaralickFeature.Entropy

alpha = 0.5 # transparency

# extract the output
collage_output = collage.get_single_feature_output(which_feature)
print(collage_output.shape)
if use_3D:
    collage_output = collage_output[:,:,94,1]

# Show preview of larger version of image.
figure = plt.figure(figsize = (15, 15))

# show the image
plt.imshow(image_array[:,:,94] if use_3D else image_array, cmap = plt.cm.gray, extent=extent)

# overlay the collage output
plt.imshow(collage_output, cmap = plt.cm.jet, alpha=alpha, extent=extent)

figure.axes[0].get_xaxis().set_visible(False)
figure.axes[0].get_yaxis().set_visible(False)

plt.title('Collage Overlay')
# %%
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm 

results = []

def blah():
    return True

with ProcessPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(blah) for i in range(1000000)]
    for future in tqdm(as_completed(futures), total=1000000):
        results.append(future.result())

print(sum(results))
# %%
