# Authors:  Roberto Mena (https://github.com/Angeluz-07) and Lawrence Chillrud (chili@u.northwestern.edu)
# Notes: Original code from Roberto Mena's helpers.py file: 
# https://github.com/Angeluz-07/MRI-preprocessing-techniques/blob/main/notebooks/helpers.py
# All functions written by Roberto Mena unless otherwise specified
 
import matplotlib.pyplot as plt
from ipywidgets import interact
import numpy as np
from ants import image_read
import SimpleITK as sitk
import cv2
import os

def plot_histogram(image, ax, n_bins=200, log=True, alpha=0.8, linewidth=3.0, show_legend=True, **kwargs):
  """
  Adapted by Lawrence Chillrud from Jacob Reinhold's intensity_normalization repo: 
  https://github.com/jcreinhold/intensity-normalization/blob/ce53748bb6b6721acad031d3b98a2e8f9921e4c6/intensity_normalization/plot/histogram.py#L212
  """
  if ax is None:
    _, ax = plt.subplots()
  data = image[image > image.mean()]
  hist, bin_edges = np.histogram(data.flatten(), n_bins)
  bins = np.diff(bin_edges) / 2 + bin_edges[:-1]
  if log:
    # catch divide by zero warnings in call to log
    hist = np.log10(hist)
    hist[np.isinf(hist)] = 0.0
  ax.plot(bins, hist, alpha=alpha, linewidth=linewidth, **kwargs)
  if show_legend: ax.legend()
  return ax

def get_scan_dict(data_dir, dir_of_interest=''):
    """
    Author: Lawrence Chillrud

    When data_dir is data/preprocessing/output/>=2, then dir_of_interest should be '', 
    otherwise, it should be 'ready_for_preprocessing' or 'ask_virginia'
    """
    scan_counts = {}
    for subject in lsdir(data_dir):
      for session in lsdir(f'{data_dir}/{subject}'):
        for scan in lsdir(f'{data_dir}/{subject}/{session}/{dir_of_interest}'):
          scan_type = scan.split('-')[1]
          if scan_type in scan_counts:
            scan_counts[scan_type] += 1
          else:
            scan_counts[scan_type] = 1
    return scan_counts

def setup():
    """
    Ensures that the current working directory is the Meningioma directory.
    Author: Lawrence Chillrud
    """
    if not os.getcwd().endswith('Meningioma'): os.chdir('../..')
    if not os.getcwd().endswith('Meningioma'): 
        raise Exception('Please run this script from the Menigioma directory')

def lsdir(path):
    """
    Returns a list of all the immediate subdirectories of a given path.
    Author: Lawrence Chillrud
    """
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

def read_example_mri(data_dir='data/preprocessing/output/2_NIFTI', subject='6', session='6_Brainlab', scan='12-AX_3D_T1_POST', ants=True, orientation='IAL'):
    """
    The function reads an MRI image file using either the ANTs or SimpleITK library, depending on the
    value of the `ants` parameter.
    
    Author: Lawrence Chillrud

    Parameters
    ----------
    data_dir : 
        The directory where the MRI data is stored.
    subject : 
        The specific subject or patient for whom the MRI scan was performed.
    session : 
        The session of the subject.
    scan : 
        The specific MRI scan that you want to read.
    ants : 
        A boolean flag determining whether to use ANTsPy or SimpleITK to read the image.
    
    Returns
    -------
    The MRI image as a SimpleITK or ANTsPy image object.
    """
    filepath = f'{data_dir}/{subject}/{session}/{scan}/{session}_{scan}.nii.gz'
    if ants: 
        return image_read(filepath, reorient=orientation)
    else:
        return sitk.ReadImage(filepath)

def explore_3D_array(arr: np.ndarray, cmap: str = 'gray'):
  """
  Given a 3D array with shape (Z,X,Y) This function will create an interactive
  widget to check out all the 2D arrays with shape (X,Y) inside the 3D array. 
  The purpose of this function to visual inspect the 2D arrays in the image. 

  Args:
    arr : 3D array with shape (Z,X,Y) that represents the volume of a MRI image
    cmap : Which color map use to plot the slices in matplotlib.pyplot
  """

  def fn(SLICE):
    plt.figure(figsize=(7,7))
    plt.imshow(arr[SLICE, :, :], cmap=cmap)

  interact(fn, SLICE=(0, arr.shape[0]-1))


def explore_3D_array_comparison(arr_before: np.ndarray, arr_after: np.ndarray, cmap: str = 'gray', title: str = 'MRI Image Comparison'):
  """
  Authors: Roberto Mena, with modifications by Lawrence Chillrud

  Given two 3D arrays with shape (Z,X,Y) This function will create an interactive
  widget to check out all the 2D arrays with shape (X,Y) inside the 3D arrays.
  The purpose of this function to visual compare the 2D arrays after some transformation. 

  Args:
    arr_before : 3D array with shape (Z,X,Y) that represents the volume of a MRI image, before any transform
    arr_after : 3D array with shape (Z,X,Y) that represents the volume of a MRI image, after some transform    
    cmap : Which color map use to plot the slices in matplotlib.pyplot
  """

  assert arr_after.shape == arr_before.shape

  def fn(SLICE):
    
    global_min = min(arr_before[:, :, SLICE].min(), arr_after[:, :, SLICE].min())
    global_max = max(arr_before[:, :, SLICE].max(), arr_after[:, :, SLICE].max())

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(10,10))
    fig.suptitle(title, fontsize=18, y=0.80)

    ax1.set_title('Before', fontsize=15)
    im1 = ax1.imshow(arr_before[:, :, SLICE], cmap=cmap, vmin=global_min, vmax=global_max)
    fig.colorbar(im1, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)

    ax2.set_title('After', fontsize=15)
    im2 = ax2.imshow(arr_after[:, :, SLICE], cmap=cmap, vmin=global_min, vmax=global_max)
    fig.colorbar(im2, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout()
  
  interact(fn, SLICE=(0, arr_before.shape[2]-1))


def show_sitk_img_info(img: sitk.Image):
  """
  Given a sitk.Image instance prints the information about the MRI image contained.

  Args:
    img : instance of the sitk.Image to check out
  """
  pixel_type = img.GetPixelIDTypeAsString()
  origin = img.GetOrigin()
  dimensions = img.GetSize()
  spacing = img.GetSpacing()
  direction = img.GetDirection()

  info = {'Pixel Type' : pixel_type, 'Dimensions': dimensions, 'Spacing': spacing, 'Origin': origin,  'Direction' : direction}
  for k,v in info.items():
    print(f' {k} : {v}')


def add_suffix_to_filename(filename: str, suffix:str) -> str:
  """
  Takes a NIfTI filename and appends a suffix.

  Args:
      filename : NIfTI filename
      suffix : suffix to append

  Returns:
      str : filename after append the suffix
  """
  if filename.endswith('.nii'):
      result = filename.replace('.nii', f'_{suffix}.nii')
      return result
  elif filename.endswith('.nii.gz'):
      result = filename.replace('.nii.gz', f'_{suffix}.nii.gz')
      return result
  else:
      raise RuntimeError('filename with unknown extension')


def rescale_linear(array: np.ndarray, new_min: int, new_max: int):
  """Rescale an array linearly."""
  minimum, maximum = np.min(array), np.max(array)
  m = (new_max - new_min) / (maximum - minimum)
  b = new_min - m * minimum
  return m * array + b


def inspect_contours(data_dir='data/preprocessing/output', subject='6', session='6_Brainlab', scan='12-AX_3D_T1_POST', cmap='gray', orientation='RSA'):
  """Author: Lawrence Chillrud"""
  image = read_example_mri(data_dir=f'{data_dir}/4_INTENSITY_STANDARDIZED', subject=subject, session=session, scan=scan, ants=True, orientation=orientation).numpy()
  mask = image_read(f'{data_dir}/5_SKULLSTRIPPED/{subject}/{session}/{scan}/brain_mask.nii.gz', reorient=orientation).numpy()
  explore_3D_array_with_mask_contour(image, mask, thickness=1, title=f'{session}/{scan}')

def explore_3D_array_with_mask_contour(arr: np.ndarray, mask: np.ndarray, thickness: int = 1, title: str = 'MRI Image with Brain Mask Contours'):
  """
  Given a 3D array with shape (Z,X,Y) This function will create an interactive
  widget to check out all the 2D arrays with shape (X,Y) inside the 3D array. The binary
  mask provided will be used to overlay contours of the region of interest over the 
  array. The purpose of this function is to visual inspect the region delimited by the mask.

  Args:
    arr : 3D array with shape (Z,X,Y) that represents the volume of a MRI image
    mask : binary mask to obtain the region of interest
  """
  assert arr.shape == mask.shape
  
  _arr = rescale_linear(arr,0,1)
  _mask = rescale_linear(mask,0,1)
  _mask = _mask.astype(np.uint8)

  def fn(SLICE):
    arr_rgb = cv2.cvtColor(_arr[SLICE, :, :], cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(_mask[SLICE, :, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    arr_with_contours = cv2.drawContours(arr_rgb, contours, -1, (0,1,0), thickness)

    plt.figure(figsize=(7,7))
    plt.imshow(arr_with_contours)
    plt.axis('off')
    plt.title(title)

  interact(fn, SLICE=(0, arr.shape[0]-1))