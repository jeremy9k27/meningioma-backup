# %%
import os
import sys
import pandas as pd
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup

setup()

def concatenate_csvs(directory):
    # Create a list to hold dataframes
    dataframes = []
    
    # Loop through each file in the specified directory
    for filename in os.listdir(directory):
        # Check if the file is a CSV
        if filename.endswith('.csv'):
            # Construct the full file path
            file_path = os.path.join(directory, filename)
            # Read the CSV file and append it to the list of dataframes
            dataframes.append(pd.read_csv(file_path))
    
    # Concatenate all dataframes in the list
    overall_df = pd.concat(dataframes, ignore_index=True)
    
    # Write the concatenated dataframe to a new CSV file in the same directory
    overall_df.to_csv(os.path.join(directory, 'overall.csv'), index=False)

concatenate_csvs('data/radiomics_results_big_binary/Chr22q_TestSize-16/Seed-22/Scaler-Standard_SMOTE-True_EvenTestSplit-True/results_summary')

# %%
