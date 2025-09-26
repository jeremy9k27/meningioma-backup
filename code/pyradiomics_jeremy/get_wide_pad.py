import pandas as pd
import itertools
import os

# Input/output files
INPUT_FILE = 'features.csv'
OUTPUT_FILE = 'wide_features_pad.csv'

# Expected combinations
expected_modalities = ['t1c', 't1n', 't2f', 't2w']
expected_scan_types = ['(1,)', '(2,)', '(3,)', '(1, 3)', '(2, 3)', '(1, 2, 3)']
expected_combinations = list(itertools.product(expected_modalities, expected_scan_types))  # 4 × 6 = 24

# Read input
print(f"Reading data from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
original_row_count = len(df)

# Set column names
patient_col = 'Patient Number'
modality_col = 'Modality'
scan_type_col = 'Segmentation Label'

# Check for and report duplicates
print("Checking for duplicates...")
dup_counts = df.groupby([patient_col, modality_col, scan_type_col]).size().reset_index(name='count')
duplicates = dup_counts[dup_counts['count'] > 1]

if not duplicates.empty:
    print(f"Found {len(duplicates)} duplicate combinations:")
    for _, row in duplicates.iterrows():
        print(f"  Patient {row[patient_col]}, Modality: {row[modality_col]}, Segmentation: {row[scan_type_col]} appears {row['count']} times")
    
    # Remove duplicates, keeping the first occurrence
    df = df.drop_duplicates(subset=[patient_col, modality_col, scan_type_col], keep='first')
    print(f"Removed {original_row_count - len(df)} duplicate rows. {len(df)} rows remaining.")
else:
    print("No duplicates found in the data.")

# List of value columns to flatten
value_cols = [col for col in df.columns if col not in []]

# Delete the output file if it exists (optional, for clean start)
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

print(f"Processing {df[patient_col].nunique()} patients...")

# Group by patient and write one row at a time
for idx, (pid, group) in enumerate(df.groupby(patient_col)):
    # Set index for joining
    group_indexed = group.set_index([modality_col, scan_type_col])

    # Reindex to ensure all 24 combinations exist
    try:
        full_index = pd.MultiIndex.from_tuples(expected_combinations, names=[modality_col, scan_type_col])
        group_reindexed = group_indexed.reindex(full_index).reset_index()
        
        # Flatten into wide format
        flat = {}
        for i, row in group_reindexed.iterrows():
            slice_label = f"{i+1:02d}"
            for col in value_cols:
                flat[f"{col}_{slice_label}"] = row.get(col, pd.NA)

        flat[patient_col] = pid

        wide_row = pd.DataFrame([flat])

        # Append to file
        if idx == 0:
            wide_row.to_csv(OUTPUT_FILE, index=False, mode='w', header=True)
        else:
            wide_row.to_csv(OUTPUT_FILE, index=False, mode='a', header=False)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1} patients...")
            
    except ValueError as e:
        # If there's still an error, print detailed info about this patient's data
        print(f"Error processing patient {pid}: {e}")
        print(f"Patient {pid} data sample:")
        print(group.head())
        print(f"Duplicate check for this patient:")
        dup_check = group.duplicated(subset=[modality_col, scan_type_col], keep=False)
        if dup_check.any():
            print("DUPLICATES STILL EXIST IN THIS PATIENT'S DATA!")
            print(group[dup_check].sort_values(by=[modality_col, scan_type_col]))
        else:
            print("No duplicates in this patient's data. Different issue.")
        continue

print(f"Processing complete. Output saved to {OUTPUT_FILE}")