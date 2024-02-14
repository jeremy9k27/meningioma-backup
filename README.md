# Meningioma
Meningioma biomarkers analysis using radiomics and deep learning

Note: The data used in this project is not publicly available to comply with regulations. Therefore I have not included any data nor any output resulting from data in this repository.

## Table of Contents
- [Preprocessing](#preprocessing)

## Preprocessing
### 0. Downloading the data:

Turn on Global Protect VPN. Go to [NURIPS](https://nuripsweb01.fsm.northwestern.edu/app/template/Index.vm) and find the "Meningiomas" project. Then under actions, pick "Download Images". Don't select all the sessions, just those with "Brainlab" in the title. Pick all DICOM scans. Use option 2 to download a zip file. This should be roughly 11GB large. Do NOT "simplify the download structure", as it throws away the names of the scans when "simplifying".

### 1. Scan Type Cleanup: 

Now the really annoying part... Identifying the scans that we want to use from the Brainlab sessions, and standardizing the names. This all unfortunately needs to be done by hand, due to all the inconsistencies in how the sequences were originally labelled. Script [1a_thumbnails.py](code/preprocessing/1a_thumbnails.py) was used to generate thumbnails for easy visual inspection of scans, so radiologists could lend me their expertise in coming up with a consistent and rational labeling scheme. Script [1b_remove_b0_from_tracews.py](code/preprocessing/1b_remove_b0_from_tracews.py) ensured that all scans labelled `AX_DIFFUSION` contained B1000 images, since some of the `DIFFUSION_TRACEW` sequences that were labelled as `AX_DIFFUSION` had B0 scans in them that were not wanted. Script [1c_check_scan_names.py](code/preprocessing/1c_check_scan_names.py) was used to get a final overview of the scan types and their counts. The scan types of interest (after relabelling became): `AX_2D_T2`, `AX_3D_T1_POST`, `AX_3D_T1_PRE`, `AX_ADC`, `AX_DIFFUSION` (all of these are B1000 images), `AX_SWI`, `SAG_3D_FLAIR`, `SAG_3D_FLAIR_POST`, `SAG_3D_T1_POST`, `SAG_3D_T2`. The scan counts we are working with as of 2/13/24 across 80 subjects in our cohort are:

```
{'AX_2D_T2': 1,
 'AX_3D_T1_POST': 83,
 'AX_3D_T1_PRE': 24,
 'AX_ADC': 77,
 'AX_DIFFUSION': 77,
 'AX_SWI': 20,
 'SAG_3D_FLAIR': 77,
 'SAG_3D_T2': 27}
```

### 2. DICOM to NIFTI conversion: 

The script [2a_convert_dicoms_to_nifti.py](code/preprocessing/2a_convert_dicoms_to_nifti.py) does what it sounds like. It produces a log file whose warnings can be explored using script [2b](code/preprocessing/2b_parse_logfile_warnings.py). Script [2a](code/preprocessing/2a_convert_dicoms_to_nifti.py) uses Chris Rorden's [dcm2niiX](https://github.com/rordenlab/dcm2niix) command line program. See below for the exact executable you will need, as the default doesn't work for all scans in our Meningioma cohort due to DICOM formatting inconsistencies across some scans. _Some warnings in the resulting logfile to go through:_

    a. `Warning: Unsupported transfer syntax '1.2.840.10008.1.2.4.90'`: This warning should only come up if you have the default dcm2niiX installed through e.g., miniconda. It should not appear if you have the `JP2:OpenJPEG` build (other transfer syntaxes may still be unsupported and will yield warnings accordingly). As explained in the dcm2niiX [documentation](https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage#DICOM_Transfer_Syntaxes_and_Compressed_Images), this offending transfer syntax corresponds to the `JPEG2000-lossless` format. This is an old (and rarely used) format that by default dcm2niiX does not support. Luckily for us, it does have the _option_ of supporting this format, we just need to download [the correct version](https://github.com/rordenlab/dcm2niix/releases/download/v1.0.20230411/dcm2niix_macos.zip). Once downloaded and unzipped, this executable needs to be placed in the [code/preprocessing](code/preprocessing/) folder for the script to run correctly. I've also added the executable to the [.gitignore](.gitignore) so it doesn't clutter up the repo.
    
    b. `Warning: Siemens XA exported as classic not enhanced DICOM (issue 236)`: We can probably ignore this warning. Visually inspect some of the .nii.gz files that suffer this warning to double check. See [issue 236](https://github.com/rordenlab/dcm2niix/issues/236) for details.

    c. `Warning: Siemens MoCo? Bogus slice timing`: We can probably ignore this warning. Visually inspect some of the .nii.gz files that suffer this warning to double check.

At the end of this step the dataset of `.nii.gz` files produced is roughly 2.6GB large.

### 3. N4 Bias Field Correction:

N4 bias field correction is handled with the [3a_parallel_n4_correction.py](code/preprocessing/3a_parallel_n4_correction.py) script, which makes use of the [ANTsPy](https://github.com/ANTsX/ANTsPy) library for the bias field correction. I use default ANTsPy parameters since this is a fairly vanilla preprocessing step. The preamble documentation in script [3a](code/preprocessing/3a_parallel_n4_correction.py) contains all the relevant details. 

Script [3b_n4_results_inspection.py](code/preprocessing/3b_n4_results_inspection.py) then allows you to view the results of the bias field correction in detail, slice by slice. Note this last time around I did not save the bias fields themselves since they take up quite a bit of disk space.

At this point, I addressed the bizarre shape of subject 5's AX_DIFFUSION scan (it was 4 dimensional instead of 3, with the last dimension having 2 elements, suggesting 2 volumes were stacked on top of one another). Script [3c_fix_weird_scans.py](code/preprocessing/3c_fix_weird_scans.py) was used to fix this issue, extracting the better of the two scans and overwriting the n4 corrected scan so it would be the proper shape. This could've been done earlier, but dicom -> nifti conversion and n4 correction worked fine on the weird scan, so it's alright to handle this here, at this point in the pipeline. For the following steps, though, this should definitely be taken care of now. 

At the end of this step the dataset of `.nii.gz` files produced is roughly 4.8GB large.

### 4. Skullstripping:

Because meningiomas can invade the skull of the patient, we don't want to skull strip just yet, as we'd potentially lose quite a bit of the tumor content in doing so (this was verified empirically during my first attempt). Instead, later down the line for the biomarker prediction task, we can skullstrip the registered images to get the less-than-perfect brain masks, and take the union of those volumetric masks with those segmentation masks produced by expert clinical radiologists, who segmented the tumor on the non-skull stripped (registered) images. Then using both of these masks, we can obtain everything we want (brain + full tumor tissue) for the downstream predictive models. Note that this approach won't work for a segmentation task, but is great for our biomarker prediction project.

So, then, why are we skullstripping now? Because we'd like these rough brain masks both for the pixel intensity normalization and registration steps to follow. Even though these brain masks won't be perfect, they will still help us considerably with these next two steps in the preprocessing pipeline.

To that end, I use FreeSurfer's [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) tool. To use SynthStrip, we need to first install [Docker](https://docs.docker.com/get-docker/) (and open the Desktop client everytime you want to run skullstripping). Then we can download the SynthStrip docker executable with:

`curl -O https://raw.githubusercontent.com/freesurfer/freesurfer/dev/mri_synthstrip/synthstrip-docker && chmod +x synthstrip-docker`

Make sure the executable is placed in the [code/preprocessing](code/preprocessing/) folder for the [4a_skullstrip.py](code/preprocessing/4a_skullstrip.py) script to run correctly. Then run the executable once to download the default model (this only has to be done once) with: 

```
cd code/preprocessing # if you're not already in this folder
./synthstrip-docker --help
```

The script [4a_skullstrip.py](code/preprocessing/4a_skullstrip.py) then runs the SynthStrip tool on all the scans to extract the brain tissue.

_Important Warning: You will probably need to go into your Docker Desktop client settings and increase the available resources to your Docker setup. If you don't do this, it's likely you could encounter OOM errors durring skullstrips._

This step produces another 1.8GB of `.nii.gz` files (the skull stripped images along with their brain masks).

### 5. Intensity Standardization:

This was a step that went through a few iterations before settling on a method. The overall idea is we'd like to preserve each pulse sequence's defining characteristics, while standardizing across subjects for better comparability between subjects. 

#### What did not work:

I started out trying [piecewise linear histogram matching](https://intensity-normalization.readthedocs.io/en/latest/algorithm.html#piecewise-linear-histogram-matching-nyul-udupa) - proposed by [Nyul, Upuda and Zhang](https://ieeexplore.ieee.org/abstract/document/836373?casa_token=DHiN18xB-fIAAAAA:-loy9cE_BOsGlNQ3kH_SOnzM2-za0hJjpsyi2h2w7Kd7ZAYv-70qHxqZVTVvWfmFMRakpWgmOA) - via Jacob Reinhold's [intensity-normalization](https://github.com/jcreinhold/intensity-normalization) package. You can see some helpful examples using this package [here](https://intensity-normalization.readthedocs.io/en/latest/usage.html#python-api-for-normalization-methods). [My code](code/preprocessing/deprecated/4a_intensity_standardization.py) to carry out piecewise linear histogram matching can be found in the [code/preprocessing/deprecated](code/preprocessing/deprecated/) folder. The problem with this approach was that we found it introduced some artificial spikes in the histogram distributions of the pixel intensities of each image. So next we tried histogram equalization.

Histogram equalization matches the histograms of each image to a uniform distribution. This drastically increased the contrast in each image, and rescaled the pixel intensity range to be from [0, 1]. This was too poor a resolution to be able to easily makeout visual features important for segmentation, e.g., tumor texture. I could have played around with the clip limit, e.g. rescale to [0, 2] and see if that offers better resolution / detail, but it looked like it was going to be too painful to tweak this by hand for each set of scans. Another idea was use an adaptive histogram equalization approach with a relatively large kernel for a smoother image. But this still begged the question of how to play with clip limits for each imaging modality. The other trouble with this approach was that it did not take into account some pulse sequences idiosyncracies. E.g., ADC shouldn't be equalized at all as that would change the relative pixel values, which we need to preserve for their semantic content. Again, [the code](code/preprocessing/deprecated/4c_histogram_equalization.py) for this attempt can be found in the deprecated [folder](code/preprocessing/deprecated/).

#### What worked:

In the end, scan-specific z-score normalization for each pulse sequence in our dataset worked best to align the histograms. I implemented the approach described [here](https://intensity-normalization.readthedocs.io/en/latest/algorithm.html#z-score) in the script [5a_zscore_normalize.py](code/preprocessing/5a_zscore_normalize.py), which relies on the brain masks produced in the previous step to calculate the sample statistics (mean and stddev inside the brain mask region). Every scan was z-score normalized using their own mean and standard deviations, _except for the ADC scans._ Since we wanted to preserve the relative intensities of the pixel values for the ADC scans, we did a global z-score normalization, using the global mean and stddev from all the ADC scans when normalizing each individual scan. Again the brain masks are used so we don't get thrown off by signals coming from the skull. 

Script [5b_view_intensity_normalization.py](code/preprocessing/5b_view_intensity_normalization.py) then allows you to view the results of the intensity normalization in a more hands on fashion, beyond the simple before and after histograms that script [5a](code/preprocessing/5a_zscore_normalize.py) spits out.

This step produced 9.8GB of `.nii.gz` files.

_An important warning: The first time I did this, I forgot to shift the z-scored histograms so that they would start at 0 rather than a negative number. This caused headaches during registration, and has since been corrected._

### 6. Registration:

Registration is done in script [6a_parallel_registration_w_propogation.py](code/preprocessing/6a_parallel_registration_w_propogation.py), which saves all intermediate registrations, the transform `.mat` files used, and of course, the final registered images. The ANTsPy library is again used for registration. The protocal *for each subject* is as follows:

1. Register the T1 post to the MNI template using an affine registration (12 deg. of freedom, rotations, translations, and scaling) and save that transformation for propogating it to other scans later.
2. Register non-diffusion type images to the T1 post using an affine registration, save those transforms. Then, propogate the T1 post -> MNI transform from step 1 onto all these scans.
3. Now for the diffusion type images (SWI, DIFFUSION, ADC). These require some extra love and care due to the differences in their presentation from the other scans. First we affine register each of these to the FLAIR. Then we propogate the FLAIR -> T1 post transform onto each, and then again propogate the T1 post -> MNI onto each scan. For the ADC, note that it uses the propogated transforms from the DIFFUSION -> FLAIR, rather than making its own affine transform to the FLAIR, since it is the same as the DIFFUSION just postprocessed by the radiologist who aqcuired the image.

The script [6b_view_propogated_registration.py](code/preprocessing/6b_view_propogated_registration.py) allows you to view this whole process step by step.

This step produced 18.8GB of `.nii.gz` files.

### 7. Wrapping it all up for segmentation

Scripts [7a](code/preprocessing/7a_copy_preprocessed_data.py) and [7b](code/preprocessing/7b_view_final_scans.py) then copy over the completed preprocessed scans for delivery to NURIPS. On the project page on NURIPS, go to Actions -> Manage Files -> Upload Files to upload zips of the preprocessed scans. 

The dataset that got uploaded to NURIPS was roughly 9GB large total.