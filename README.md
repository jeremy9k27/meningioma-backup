# Meningioma
Meningioma biomarkers analysis using radiomics and deep learning

## Table of Contents
- [Preprocessing](#preprocessing)

## Preprocessing
0. **Downloading the data:** Turn on Global Protect VPN. Go to [NURIPS](https://nuripsweb01.fsm.northwestern.edu/app/template/Index.vm) and find the "Meningiomas" project. Then under actions, pick "Download Images". Don't select all the sessions, just those with "Brainlab" in the title. Pick all DICOM scans. Use option 2 to download a zip file. This should be roughly 11GB large. Do NOT "simplify the download structure", as it throws away the names of the scans when "simplifying".

1. **Scan Type Cleanup:** Now the really annoying part... Identifying the scans that we want to use from the Brainlab sessions, and standardizing the names. This all unfortunately needs to be done by hand, due to all the inconsistencies in how the sequences were originally labelled. A first attempt at automating this can be seen in scripts [1a](code/preprocessing/1a_original_scan_type_cleanup_labels.R) and [1b](code/preprocessing/1b_scan_type_label_mapping.R), but they were still not perfect, so relabelling had to be done by hand. To aid in making sure labels were properly assigned, script [1c](code/preprocessing/1c_thumbnails.py) was written to generate thumbnails for easy visual inspection of scans. The scan types of interest (after relabelling became): `AX_2D_T2`, `AX_3D_T1_POST`, `AX_3D_T1_PRE`, `AX_ADC`, `AX_DIFFUSION` (all of these are B1000 images), `AX_SWI`, `SAG_3D_FLAIR`, `SAG_3D_FLAIR_POST`, `SAG_3D_T1_POST`, `SAG_3D_T2`. Script [1d](code/preprocessing/1d_remove_b0_from_tracews.py) ensured that all scans labelled `AX_DIFFUSION` contained B1000 images, since some of the `DIFFUSION_TRACEW` sequences that were labelled as `AX_DIFFUSION` had B0 scans in them that were not wanted. Finally, script [1e](code/preprocessing/1e_check_scan_names.py) was used to get a final overview of the scan types and their counts.

2. **DICOM to NIFTI conversion:** The script [2a_convert_dicoms_to_nifti.py](code/preprocessing/2a_convert_dicoms_to_nifti.py) does what it sounds like. It produces a log file whose warnings can be explored using script [2b](code/preprocessing/2b_parse_logfile_warnings.py). The final dataset of `.nii.gz` files is roughly 2.5GB large. Script [2a](code/preprocessing/2a_convert_dicoms_to_nifti.py) uses Chris Rorden's [dcm2niiX](https://github.com/rordenlab/dcm2niix) command line program. See below for the exact executable you will need, as the default doesn't work for all scans in our Meningioma cohort due to DICOM formatting inconsistencies across some scans. _Some warnings in the resulting logfile to go through:_

    a. `Warning: Unsupported transfer syntax '1.2.840.10008.1.2.4.90'`: This warning should only come up if you have the default dcm2niiX installed through e.g., miniconda. It should not appear if you have the `JP2:OpenJPEG` build (other transfer syntaxes may still be unsupported and will yield warnings accordingly). As explained in the dcm2niiX [documentation](https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage#DICOM_Transfer_Syntaxes_and_Compressed_Images), this offending transfer syntax corresponds to the `JPEG2000-lossless` format. This is an old (and rarely used) format that by default dcm2niiX does not support. Luckily for us, it does have the _option_ of supporting this format, we just need to download [the correct version](https://github.com/rordenlab/dcm2niix/releases/download/v1.0.20230411/dcm2niix_macos.zip). Once downloaded and unzipped, this executable needs to be placed in the [code/preprocessing](code/preprocessing/) folder for the script to run correctly. I've also added the executable to the [.gitignore](.gitignore) so it doesn't clutter up the repo.
    
    b. `Warning: Siemens XA exported as classic not enhanced DICOM (issue 236)`: We can probably ignore this warning. Visually inspect some of the .nii.gz files that suffer this warning to double check. See [issue 236](https://github.com/rordenlab/dcm2niix/issues/236) for details.

    c. `Warning: Siemens MoCo? Bogus slice timing`: We can probably ignore this warning. Visually inspect some of the .nii.gz files that suffer this warning to double check.

    d. Problematic scan: 40_Brainlab/107-SAG_3D_T1_POST

        Warning: Instance Number (0020,0013) order is not spatial.
        Warning: Interslice distance varies in this volume (incompatible with NIfTI format)
        Warning: Missing images? Expected 168 images, but instance number (0020,0013) ranges from 173 to 1
        Warning: Unable to rotate 3D volume: slices not equidistant: 0.999998 != 2
