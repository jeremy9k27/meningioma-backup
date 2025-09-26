import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup
from utils import read_ndarray
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

setup()

num_workers = 16
data_dir = 'data/collage_sparse_small_windows/windowsize-5_binsize-64'
collage_feat_names = [
    'AngularSecondMoment', 'Contrast', 'Correlation',
    'SumOfSquareVariance', 'SumAverage', 'SumVariance',
    'SumEntropy', 'Entropy', 'DifferenceVariance', 'DifferenceEntropy',
    'InformationMeasureOfCorrelation1', 'InformationMeasureOfCorrelation2', 'MaximalCorrelationCoefficient'
]

def get_metadata_from_filename(filename):
    # get sequence type
    if 'FLAIR' in filename:
        sequence = 'FLAIR'
    elif 'T1' in filename:
        sequence = 'T1'
    elif 'ADC' in filename:
        sequence = 'ADC'
    else:
        raise ValueError('Unknown sequence type! Not one of FLAIR, T1, ADC.')

    # get subject number
    sub_no = filename.split('subject-')[-1].split('_')[0]

    # get segmentation label
    seg_label = filename.split('seg-')[-1].split('.joblib')[0]
    
    return int(sub_no), sequence, seg_label

def calculate_summary_stats(arr, prefix, sub_no):
    mask = ~np.isnan(arr)
    hist, bin_edges = np.histogram(arr[mask], bins=10, density=True)
    probabilities = hist * np.diff(bin_edges)

    uniformity = np.sum(hist**2 * np.diff(bin_edges))

    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)

    mean_val = np.nanmean(arr)
    std_val = np.nanstd(arr)
    energy = np.nansum(arr**2)
    volume = mask.sum()

    arr_90 = np.nanpercentile(arr, 90)
    arr_10 = np.nanpercentile(arr, 10)
    robust_arr = arr[(arr >= arr_10) & (arr <= arr_90)]
    robust_mean_abs_dev = np.mean(np.abs(robust_arr - np.mean(robust_arr)))

    return pd.DataFrame({
        'Subject Number': sub_no,
        f'{prefix}_mean': mean_val,
        f'{prefix}_std': std_val,
        f'{prefix}_var': std_val**2,
        f'{prefix}_min': min_val,
        f'{prefix}_max': max_val,
        f'{prefix}_median': np.nanmedian(arr),
        f'{prefix}_iqr': np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25),
        f'{prefix}_range': max_val - min_val,
        f'{prefix}_10perc': arr_10,
        f'{prefix}_90perc': arr_90,
        f'{prefix}_energy': energy,
        f'{prefix}_totalenergy': volume * energy,
        f'{prefix}_MAD': np.nanmean(np.abs(arr - mean_val)),
        f'{prefix}_rMAD': robust_mean_abs_dev,
        f'{prefix}_rms': np.sqrt(energy / volume),
        f'{prefix}_uniformity': uniformity,
        f'{prefix}_skewness': skew(arr, axis=None, nan_policy='omit'),
        f'{prefix}_kurtosis': kurtosis(arr, axis=None, nan_policy='omit'),
        f'{prefix}_entropy': entropy(probabilities)
    }, index=[0])

def post_process_collage(filename):
    sub_no, sequence, seg_label = get_metadata_from_filename(filename)
    collage = read_ndarray(f'{data_dir}/{filename}')

    all_stats_df = pd.DataFrame(columns=['Subject Number'])
    for angle in range(collage.shape[-1]):
        for feat in range(collage.shape[3]):
            feat_name = collage_feat_names[feat]
            c_collage = collage[:, :, :, feat, angle]
            current_stats_df = calculate_summary_stats(c_collage, f'{sequence}-{seg_label}-{feat_name}_ang{angle}', sub_no)
            all_stats_df = pd.merge(all_stats_df, current_stats_df, on='Subject Number', how='outer')
    
    return all_stats_df

def main():
    collage_files = [f for f in os.listdir(data_dir) if f.endswith('.joblib')]
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executer:
        futures = [executer.submit(post_process_collage, f) for f in collage_files]

        for future in tqdm(as_completed(futures), total=len(futures), dynamic_ncols=True, smoothing=0):
            results.append(future.result())

    pd.concat(results, axis=0).reset_index().drop(columns=['index']).groupby('Subject Number').agg('first').reset_index().to_csv(f'{data_dir}_summary.csv', index=False)

if __name__ == '__main__':
    main()
