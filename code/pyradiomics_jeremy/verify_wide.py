import pandas as pd

# Define input and output file paths
INPUT_FILE  = 'wide_features_pad.csv'

df = pd.read_csv(INPUT_FILE)

print(f"Number of rows: {len(df)}")
print(f"Column names: {len(df.columns)}")

total_nans = df.isna().sum().sum()
print(f"Total NaN values: {total_nans}")