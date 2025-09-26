import pandas as pd

# Define input and output file paths
INPUT_FILE = 'features.csv'
OUTPUT_FILE = 'wide_features.csv'

# Read the input file
narrow_df = pd.read_csv(INPUT_FILE)


wide_rows = []

# Group consecutive rows by 'patient_id'
current_group = []
current_pid = None

for idx, row in narrow_df.iterrows():

    pid = row['Patient Number']

    if pid != current_pid:
        if len(current_group) == 24:
            # 3a. Flatten the 24 rows into one dict of {col_slice: value}
            flat = {}
            for i, grp_row in enumerate(current_group, start=1):
                slice_label = f"{i:02d}"   # zero-pad, e.g. '01', '02', …
                for col, val in grp_row.items():
                    if col == 'patient_id':
                        continue
                    flat[f"{col}_{slice_label}"] = val


            wide_series = pd.Series(flat)

            wide_rows.append(wide_series)

        # Reset for the new patient
        current_group = [row]
        current_pid   = pid

    else:
        current_group.append(row)

# Handle the final group
if len(current_group) == 24:
    flat = {}
    for i, grp_row in enumerate(current_group, start=1):
        slice_label = f"{i:02d}"
        for col, val in grp_row.items():
            if col == 'patient_id':
                continue
            flat[f"{col}_slice{slice_label}"] = val
    wide_series = pd.Series(flat)
    wide_series['patient_id'] = current_pid
    wide_rows.append(wide_series)




# Combine all wide rows into a DataFrame
if wide_rows:
    wide_df = pd.DataFrame(wide_rows)


else:
    wide_df = pd.DataFrame()


#wide_df.to_csv(OUTPUT_FILE, index=False)