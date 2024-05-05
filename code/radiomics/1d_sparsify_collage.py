# %%
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup
from utils import write_ndarray
import joblib
from tqdm import tqdm
import logging
import time
from datetime import datetime

setup()

def get_dir_size(directory):
    total_size = 0
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024 * 1024)  # Convert bytes to GB

old_dir = 'data/collage/windowsize-9_binsize-64'
new_dir = 'data/collage_sparse/windowsize-9_binsize-64'

if not os.path.exists(new_dir): os.makedirs(new_dir)

joblib_files = [f for f in os.listdir(old_dir) if f.endswith('.joblib')]
n = len(joblib_files)

logging.basicConfig(filename='data/collage_sparse/conversion_log.txt', level=logging.INFO, format='%(message)s')
overall_begin_time = time.time()
overall_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
bar = '-' * 80
logging.info(bar)
logging.info(f'Starting collage feature conversion into sparse matrices for more efficient storage for n = {n} files at {overall_start_time}')
logging.info(f'Old directory to be converted into sparse format: {old_dir}')
logging.info(f'New directory to store sparse matrices: {new_dir}')
logging.info(bar)

for fname in tqdm(joblib_files, total=n, desc='Storing collage features as sparse mats'):
    arr = joblib.load(f'{old_dir}/{fname}')
    try:
        logging.info(f'Converting {fname} to sparse matrix...')
        write_ndarray(arr, f'{new_dir}/{fname}')
        logging.info(f'\tSuccess!')
    except Exception as e:
        logging.error(f'\tError converting {fname} to sparse matrix: {e}')

overall_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
overall_time_elapsed = time.time() - overall_begin_time
hours, rem = divmod(overall_time_elapsed, 3600)
minutes, seconds = divmod(rem, 60)
overall_time_elapsed = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
logging.info(f'\n\nCompleted collage sparse conversions for all files at {overall_end_time}')
logging.info(f'Total elapsed time: {overall_time_elapsed}\n')
logging.info(f'Old directory size: {get_dir_size(old_dir):.2f} GB')
logging.info(f'New directory size: {get_dir_size(new_dir):.2f} GB')
logging.info(bar)

# %%
