# Meningioma
Meningioma biomarkers analysis using radiomics and deep learning

## Table of Contents
- [Preprocessing](#preprocessing)

## Preprocessing
1. **Downloading the data:** Turn on Global Protect VPN. Go to [NURIPS](https://nuripsweb01.fsm.northwestern.edu/app/template/Index.vm) and find the "Meningiomas" project. Then under actions, pick "Download Images". Don't select all the sessions, just those with "Brainlab" in the title. Pick all DICOM scans. Use option 2 to download a zip file. This should be roughly 11GB large. Do NOT "simplify the download structure", as it throws away the names of the scans when "simplifying".

2. **Scan Type Cleanup:** Now the really annoying part... Identifying the scans that we want to use from the Brainlab sessions, and standardizing the names. This all unfortunately needs to be done by hand, due to all the inconsistencies in how the sequences were originally labelled. A first attempt at automating this can be seen in scripts [1a](code/preprocessing/1a_original_scan_type_cleanup_labels.R) and [1b](code/preprocessing/1b_scan_type_label_mapping.R), but they were still not perfect, so relabelling had to be done by hand. To aid in making sure labels were properly assigned, script [1c](code/preprocessing/1c_thumbnails.py) was written to generate thumbnails for easy visual inspection of scans. The scan types of interest (after relabelling became): `AX_2D_T2`, `AX_3D_T1_POST`, `AX_3D_T1_PRE`, `AX_ADC`, `AX_DIFFUSION` (all of these are B1000 images), `AX_SWI`, `SAG_3D_FLAIR`, `SAG_3D_FLAIR_POST`, `SAG_3D_T1_POST`, `SAG_3D_T2`. Script [1d](code/preprocessing/1d_remove_b0_from_tracews.py) ensured that all scans labelled `AX_DIFFUSION` contained B1000 images, since some of the `DIFFUSION_TRACEW` sequences that were labelled as `AX_DIFFUSION` had B0 scans in them that were not wanted. Finally, script [1e](code/preprocessing/1e_check_scan_names.py) was used to get a final overview of the scan types and their counts.

3. **DICOM to NIFTI conversion:**