# File: 2c_scan_type_cleanup.py
# Date: 06/27/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description: Automated scan type cleanup script using metadata in accompanying json files from NIFTI conversion to classify scans.

#--------------------------#
####      CONTENTS      ####
#--------------------------#
# N. Notes
# 0. Package imports

#--------------------------#
####      N. NOTES      ####
#--------------------------#
# This script is meant to perform automated scan type cleanup using metadata in
# accompanying json files from NIFTI conversion to classify scans.
#
# This script relies on the following file(s) as inputs:
#   * data/round2_preprocessing/output/2_NIFTI/*/*/*/*.json
#
# This script generates the following file(s) as outputs:
#   *
#%%------------------------#
#### 0. PACKAGE IMPORTS ####
#--------------------------#
import pandas as pd
from utils import lsdir, setup
import json
import os

#---------------------------#
#### 1. SET UP FILEPATHS ####
#---------------------------#
setup()

data_dir = 'data/round2_preprocessing/output/2_NIFTI'

#---------------------------#
#### 2. HELPER FUNCTIONS ####
#---------------------------#
def clean_scan_name(scan_name):
    scan_name = scan_name.lower()
    if 'b1000' in scan_name or 'b=1000' in scan_name or 'tracew' in scan_name:
        return 'B1000'
    if 'adc' in scan_name:
        return 'ADC'
    if 'flair' in scan_name:
        return 'FLAIR'
    if 'diffusion' in scan_name:
        return 'DWI'
    if 'mprage' in scan_name:
        if 'post' in scan_name:
            return 'T1 POST'
        if 'pre' in scan_name:
            return 'T1 PRE'
        return 'T1'
    if 't2' in scan_name:
        return 'T2'
    if 't1' in scan_name:
        return 'T1'

    return None

def classify_scan_type(json_file):
    # Read in metadata
    with open(json_file, 'r') as file:
        data = json.load(file)

    et = None
    if 'EchoTime' in data.keys():
        et = data['EchoTime']

    rt = None
    if 'RepetitionTime' in data.keys():
        rt = data['RepetitionTime']
    
    fa = None
    if 'FlipAngle' in data.keys():
        fa = data['FlipAngle']
    
    it = None
    if 'InversionTime' in data.keys():
        it = data['InversionTime']

    slice_thickness = None
    if 'SliceThickness' in data.keys():
        slice_thickness = data['SliceThickness']
    
    # Find out which scan types the json's metadata matches
    detected = []
    
    # Slice thickness
    if slice_thickness is not None:
        if slice_thickness <= 1:
            detected.append('3D')
        else:
            detected.append('2D')

    # All four criterion needed
    if rt is not None and et is not None and fa is not None and it is not None:
        # NU MPRAGE WITHOUT CONTRAST (RT: 2.1, ET: 0.00246, IT: 1, FA: 12) 
        if rt == 2.1 and et == 0.00246 and it == 1 and fa == 12:
            detected.append('NU MPRAGE WITHOUT CONTRAST')

        # NU MPRAGE WITH CONTRAST (RT: 1.78, ET: 0.00352, IT: 1.1, FA: 15)
        if rt == 1.78 and et == 0.00352 and it == 1.1 and fa == 15:
            detected.append('NU MPRAGE WITH CONTRAST')

        # NU 3D FLAIR (RT: 5, ET: 0.383, IT: 1.8, FA: 120)
        if rt == 5 and et == 0.383 and it == 1.8 and fa == 120:
            detected.append('NU 3D FLAIR')
        
        # NU 2D FLAIR (RT: 8.5, ET: 0.094, IT: 2.44, FA: 150)
        if rt == 8.5 and et == 0.094 and it == 2.44 and fa == 150:
            detected.append('NU 2D FLAIR')
        
        # STIR (RT: >2, ET: >0.060, FA: 90-180, IT: 0.120-0.170)
        if rt > 2 and et > 0.060 and fa >= 90 and fa <= 180 and it >= 0.120 and it <= 0.170:
            detected.append('STIR')


        
    # All criterion needed except for it
    if rt is not None and et is not None and fa is not None:
        # MPRAGE (RT: 2, ET: 0.002-0.004, FA: 5-12)
        if rt == 2 and et >= 0.002 and et <= 0.004 and fa >= 5 and fa <= 12:
            detected.append('MPRAGE')
        # T1 (RT: <0.8, ET: <0.030, FA: 90)
        if rt < 0.8 and et < 0.030 and fa == 90:
            detected.append('T1')
        # NU Fat Sat COR T1 (RT: 0.55, ET: 0.012, FA: 180)
        if rt == 0.55 and et == 0.012 and fa == 180:
            detected.append('NU Fat Sat COR T1')
        # T2 (RT: >2, ET: >0.080, FA: 90)
        if rt > 2 and et > 0.080 and fa == 90:
            detected.append('T2')
        # FSE T2 (RT: >2, ET: >0.060, FA: 90)
        if rt > 2 and et > 0.060 and fa == 90:
            detected.append('FSE T2')
        # NU GRE T2 (RT: 0.839, ET: 0.0199, FA: 20)
        if rt == 0.839 and et == 0.0199 and fa == 20:
            detected.append('NU GRE T2')
        # NU SWI (RT: 0.049, ET: 0.040, FA: 15)
        if rt == 0.049 and et == 0.040 and fa == 15:
            detected.append('NU SWI')
        # SWI (RT: 0.025-0.050, ET: 0.020-0.040, FA: 15-20)
        if rt >= 0.025 and rt <= 0.050 and et >= 0.020 and et <= 0.040 and fa >= 15 and fa <= 20:
            detected.append('SWI')
        # PD (RT: >1, ET: <0.030, FA: 90)
        if rt > 1 and et < 0.030 and fa == 90:
            detected.append('PD')



    # Only et and fa are needed:
    if et is not None and fa is not None:
        # GRE T1 (ET: <0.030, FA: 70-110)
        if et < 0.030 and fa >= 70 and fa <= 110:
            detected.append('GRE T1')
        # GRE T2 (ET: <0.030, FA: 5-20)
        if et < 0.030 and fa >= 5 and fa <= 20:
            detected.append('GRE T2')


    # Only rt and et are needed:
    if rt is not None and et is not None:
        # T1 (RT: 0.5, ET: 0.014)
        if rt == 0.5 and et == 0.014:
            detected.append('T1')
    
        # T2 (RT: 4, ET: 0.090)
        if rt == 4 and et == 0.090:
            detected.append('T2')

        # NU T2 (RT: 5.66, ET: 0.099 or RT: 6.39, ET: 0.100)
        if (rt == 5.66 and et == 0.099) or (rt == 6.39 and et == 0.100):
            detected.append('NU T2')
    
        # FLAIR (RT: 9, ET: 0.114)
        if rt == 9 and et == 0.114:
            detected.append('FLAIR')

        # NU 3D FLAIR (RT: 7, ET: 0.430 or RT: 5, ET: 0.383, IT: 1.8, FA: 120)
        if rt == 7 and et == 0.430:
            detected.append('NU 3D FLAIR')
        
        # NU DWI (RT: 4.3, ET: 0.068)
        if rt == 4.3 and et == 0.068:
            detected.append('NU DWI')


    # Only fa is not needed:
    if rt is not None and et is not None and it is not None:
        # FLAIR (RT: >3, ET: >0.080, IT: 1.7-2.2)
        if rt > 3 and et > 0.080 and it >= 1.7 and it <= 2.2:
            detected.append('FLAIR')
        
    return detected, et, rt, it, fa, slice_thickness
# %%
all_detected = []
undetectable_scans = 0
confusing_scans = 0
two_detected = 0
three_detected = 0
for subject in lsdir(data_dir):
    for session in lsdir(f'{data_dir}/{subject}'):
        for scan in lsdir(f'{data_dir}/{subject}/{session}'):
            jsons = [f for f in os.listdir(f'{data_dir}/{subject}/{session}/{scan}') if f.endswith('.json')]
            scanl = scan.lower()
            if 'scout' in scanl or 'b500' in scanl or 'b=500' in scanl or 'b0' in scanl or 'b=0' in scanl or '_nd' in scanl or '_mpr_' in scanl or scanl.endswith('mpr') or 'reformat' in scanl or 'localizer' in scanl or 'loc' in scanl:
                continue
            if 'mpr' in scanl and 'mprage' not in scanl:
                continue
            
            for j in jsons:
                detected, et, rt, it, fa, st = classify_scan_type(f'{data_dir}/{subject}/{session}/{scan}/{j}')
                print(f'Sub: {subject}, Sess: {session.split("_")[-1]}, Scan: {(j.split(".json")[0]).split(f"{session}_")[-1]}\n\tMatched via name: {clean_scan_name(scanl)}\n\tMatched via json file: {detected}, ET: {et}, RT: {rt}, IT: {it}, FA: {fa}, ST: {st}\n')
                all_detected.extend(detected)
                if len(detected) == 0:
                    undetectable_scans += 1
                if len(detected) > 1:
                    confusing_scans += 1
                if len(detected) == 2:
                    two_detected += 1
                if len(detected) == 3:
                    three_detected += 1
# %%
print(pd.Series(all_detected).value_counts())
print(f'Undetectable scans: {undetectable_scans}')
print(f'Confusing scans: {confusing_scans}')
print(f'Two detected scans: {two_detected}')
print(f'Three detected scans: {three_detected}')
# %%
total_scans = pd.Series(all_detected).value_counts().sum() + undetectable_scans - two_detected - (2*three_detected)
print(f'Total scans: {total_scans}')

# %%
sub_count = 0
scan_count = 0
for subject in lsdir(data_dir):
    sub_count += 1
    for session in lsdir(f'{data_dir}/{subject}'):
        for scan in lsdir(f'{data_dir}/{subject}/{session}'):
            scan_count += 1
print(f'Total subjects: {sub_count}')
print(f'Total scans (after nifti conversion): {scan_count}')

# %%
dd2 = 'data/round2_preprocessing/NURIPS_downloads/Meningiomas_R2'
sub_count = 0
scan_count = 0
for subject in lsdir(dd2):
    sub_count += 1
    for session in lsdir(f'{dd2}/{subject}'):
        for scan in lsdir(f'{dd2}/{subject}/{session}/scans'):
            scan_count += 1
print(f'Total subjects: {sub_count}')
print(f'Total scans (before nifti conversion): {scan_count}')
# %%
