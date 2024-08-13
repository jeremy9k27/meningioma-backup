import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from classify import discard_scan, clean_scan_name, classify_scan_type
import pandas as pd
from preprocessing.utils import lsdir, setup
from ants import image_read
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm

setup()

input_dir = 'data/round2_preprocessing/output/2_NIFTI'
output_dir = 'data/round2_preprocessing/SCAN_TYPE_CLEANUP'
thumbnails_dir = f'code/scan_type_cleanup_ui/static/thumbnails'
if not os.path.exists(output_dir): os.makedirs(output_dir)
if not os.path.exists(thumbnails_dir): os.makedirs(thumbnails_dir)

def get_scan_shape(scan_path):
    if not os.path.exists(scan_path):
        warnings.warn(f'{scan_path} could not be found (likely due to naming inconsistency from NIFTI converter or NURIPS), so we cannot ascertain the shape!')
        return None
    scan = image_read(scan_path, reorient='IAL').numpy()
    return scan.shape

def write_thumbnail(scan_path, overwrite=False):
    title = scan_path.split('/')[-1].replace('.nii.gz', '')
    im_path = f'{thumbnails_dir}/{title}.png'

    if not overwrite and os.path.exists(im_path):
        return im_path

    if not os.path.exists(scan_path):
        warnings.warn(f'{scan_path} could not be found (likely due to naming inconsistency from NIFTI converter or NURIPS), so a thumbnail will not be generated!')
        return None
    
    scan = image_read(scan_path, reorient='IAL').numpy()
    if len(scan.shape) != 3:
        warnings.warn(f'{scan_path} has {len(scan.shape)} dimension(s), so a thumbnail will not be generated!')
        return None
    
    slice = scan.shape[0] // 2
    plt.imshow(scan[slice, :, :], cmap='gray')
    plt.axis('off')
    plt.title(f'{title}\nSlice {slice}/{scan.shape[0]} w/shape {scan.shape[1:]}')
    plt.savefig(im_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return im_path

def automated_first_pass():
    automated = []
    needs_handcheck = []
    discarded_list = []

    # loop through all subjects, sessions, and scans, 
    # cleaning up the scan type either in an automated way, 
    # or sending it to a handcheck list for a human to examine
    for subject in tqdm(lsdir(input_dir), desc='Subjects', total=len(lsdir(input_dir)), smoothing=0, dynamic_ncols=True):
        for session in lsdir(f'{input_dir}/{subject}'):
            for scan in lsdir(f'{input_dir}/{subject}/{session}'):
                scan_id = f'{input_dir}/{subject}/{session}/{scan}'
                given_name = scan.split('-')[-1]
                scanl = scan.lower()

                # Discard any scans we don't care about, judged by their name in the discard_scan function
                if discard_scan(scanl):
                    discarded_list.append({'id': scan_id, 'given_name': given_name})
                    continue
                
                scan_path = f'{input_dir}/{subject}/{session}/{scan}/{session}_{scan}'
                cleaned_name = clean_scan_name(scanl, f'{scan_path}.json')

                if cleaned_name is not None:
                    # Here we can automatically classify the scan type by name alone
                    automated.append({'id': scan_id, 'given_name': given_name, 'clean_name': cleaned_name})
                else:
                    # Here we need to handcheck the scan type by sending it to the list that will be given to the GUI
                    # needs_handcheck.append({'id': scan_id, 'image_path': impath, 'text': stats_from_json})
                    image_path = write_thumbnail(f'{scan_path}.nii.gz')
                    json_stats = classify_scan_type(f'{scan_path}.json')
                    scan_shape = get_scan_shape(f'{scan_path}.nii.gz')
                    text = f'Subject: {subject}\nSession: {session}\nScan: {scan}\nShape: {scan_shape}\n{json_stats}'
                    needs_handcheck.append({'id': scan_id, 'image_path': image_path, 'text': text})

    # Save lists
    pd.DataFrame(discarded_list).to_csv(f'{output_dir}/discarded.csv', index=False)
    pd.DataFrame(automated).to_csv(f'{output_dir}/automated.csv', index=False)
    pd.DataFrame(needs_handcheck).to_csv(f'{output_dir}/needs_handcheck.csv', index=False)

if __name__ == "__main__":
    automated_first_pass()
