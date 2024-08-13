import os
import json
import warnings

def discard_scan(scan_name):
    scanl = scan_name.lower()
    if 'scout' in scanl or 'b500' in scanl or 'b=500' in scanl or 'b_500' in scanl or 'b_0' in scanl or 'b0' in scanl or 'b=0' in scanl or '_nd' in scanl or '_mpr_' in scanl or scanl.endswith('mpr') or 'reformat' in scanl or 'rfmt' in scanl or 'localizer' in scanl or 'loc' in scanl:
        return True
    if 'mpr' in scanl and 'mprage' not in scanl:
        return True
    return False

def clean_scan_name(scan_name, json_file):
    scan_name = scan_name.lower()

    # get the direction from the scan name
    direction = None
    if 'ax' in scan_name:
        direction = 'AX'
    elif 'sag' in scan_name:
        direction = 'SAG'
    elif 'cor' in scan_name:
        direction = 'COR'
    
    # get the dimensionality from the json file
    dimensionality = None

    if not os.path.exists(json_file):
        warnings.warn(f'File {json_file} could not be found (likely due to naming inconsistency by NIFTI converter or by NURIPS), therefore the dimensionality of the scan could not be determined!')
        data = {}
    else:
        with open(json_file, 'r') as file:
            data = json.load(file)

    slice_thickness = None
    if 'SliceThickness' in data.keys():
        slice_thickness = data['SliceThickness']
    
    if slice_thickness is not None:
        if slice_thickness <= 1:
            dimensionality = '3D'
        else:
            dimensionality = '2D'
    
    # get the scan type from the scan name
    clean_names = []
    if 'b1000' in scan_name or 'b=1000' in scan_name or 'b_1000' in scan_name or 'tracew' in scan_name:
        clean_names.append('DIFFUSION')
    if 'diffusion' in scan_name and 'adc' not in scan_name:
        clean_names.append('DIFFUSION')
    if 'adc' in scan_name:
        clean_names.append('ADC')
    if 'flair' in scan_name:
        clean_names.append('FLAIR')
    if 'mprage' in scan_name:
        if 'post' in scan_name:
            clean_names.append('T1_POST')
        elif 'pre' in scan_name:
            clean_names.append('T1_PRE')
        else:
            clean_names.append('T1')
    if 't2' in scan_name and 'gre' not in scan_name:
        clean_names.append('T2')
    if 't1' in scan_name and 'mprage' not in scan_name:
        if 'post' in scan_name:
            clean_names.append('T1_POST')
        elif 'pre' in scan_name:
            clean_names.append('T1_PRE')
        else:
            clean_names.append('T1')
    if 'swi' in scan_name:
        clean_names.append('SWI')
    if 'stir' in scan_name:
        clean_names.append('STIR')
    if 'gre' in scan_name:
        clean_names.append('GRE')
    
    clean_names = list(set(clean_names))
    single_clean_name = None
    if len(clean_names) == 1:
        single_clean_name = clean_names[0]
    
    # combine the direction, dimensionality, and scan type (as appropriate) and return
    if single_clean_name is None:
        return None
    elif single_clean_name in ['ADC', 'DIFFUSION', 'SWI', 'STIR', 'GRE']:
        if direction is not None:
            return '_'.join([direction, single_clean_name])
        return single_clean_name
    else:
        if direction is not None and dimensionality is not None:
            return '_'.join([direction, dimensionality, single_clean_name])
        elif direction is not None:
            return '_'.join([direction, single_clean_name])
        elif dimensionality is not None:
            return '_'.join([dimensionality, single_clean_name])
        else:
            return single_clean_name

def classify_scan_type(json_file):
    # Read in metadata
    if not os.path.exists(json_file):
        warnings.warn(f'File {json_file} could not be found (likely due to naming inconsistency by NIFTI converter or by NURIPS), therefore the dimensionality of the scan could not be determined!')
        return None
    
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
            detected.append('T1 PRE')

        # NU MPRAGE WITH CONTRAST (RT: 1.78, ET: 0.00352, IT: 1.1, FA: 15)
        if rt == 1.78 and et == 0.00352 and it == 1.1 and fa == 15:
            detected.append('T1 POST')

        # NU 3D FLAIR (RT: 5, ET: 0.383, IT: 1.8, FA: 120)
        if rt == 5 and et == 0.383 and it == 1.8 and fa == 120:
            detected.append('FLAIR')
        
        # NU 2D FLAIR (RT: 8.5, ET: 0.094, IT: 2.44, FA: 150)
        if rt == 8.5 and et == 0.094 and it == 2.44 and fa == 150:
            detected.append('FLAIR')
        
        # STIR (RT: >2, ET: >0.060, FA: 90-180, IT: 0.120-0.170)
        if rt > 2 and et > 0.060 and fa >= 90 and fa <= 180 and it >= 0.120 and it <= 0.170:
            detected.append('STIR')

    # All criterion needed except for it
    if rt is not None and et is not None and fa is not None:
        # MPRAGE (RT: 2, ET: 0.002-0.004, FA: 5-12)
        if rt == 2 and et >= 0.002 and et <= 0.004 and fa >= 5 and fa <= 12:
            detected.append('T1')
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
            detected.append('T2')
        # NU GRE T2 (RT: 0.839, ET: 0.0199, FA: 20)
        if rt == 0.839 and et == 0.0199 and fa == 20:
            detected.append('T2')
        # NU SWI (RT: 0.049, ET: 0.040, FA: 15)
        if rt == 0.049 and et == 0.040 and fa == 15:
            detected.append('SWI')
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
            detected.append('T1')
        # GRE T2 (ET: <0.030, FA: 5-20)
        if et < 0.030 and fa >= 5 and fa <= 20:
            detected.append('T2')

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
            detected.append('T2')
    
        # FLAIR (RT: 9, ET: 0.114)
        if rt == 9 and et == 0.114:
            detected.append('FLAIR')

        # NU 3D FLAIR (RT: 7, ET: 0.430 or RT: 5, ET: 0.383, IT: 1.8, FA: 120)
        if rt == 7 and et == 0.430:
            detected.append('FLAIR')
        
        # NU DWI (RT: 4.3, ET: 0.068)
        if rt == 4.3 and et == 0.068:
            detected.append('DIFFUSION')

    # Only fa is not needed:
    if rt is not None and et is not None and it is not None:
        # FLAIR (RT: >3, ET: >0.080, IT: 1.7-2.2)
        if rt > 3 and et > 0.080 and it >= 1.7 and it <= 2.2:
            detected.append('FLAIR')
    
    stats = f"The .json file suggests: {sorted(list(set(detected)))}\n\tEcho Time: {et}\n\tRepetition Time: {rt}\n\tInversion Time: {it}\n\tFlip Angle: {fa}\n\tSlice Thickness: {slice_thickness}"
    return stats