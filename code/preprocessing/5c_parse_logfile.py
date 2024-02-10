# File: 5c_parse_logfile.py
# Date: 01/24/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description: Helps declutter the logfile from the skull stripping step.

#--------------------------#
####      N. NOTES      ####
#--------------------------#
# This script is meant to help declutter the logfile from the skull stripping step, extracting only the important warnings and errors.
# By running something like python 5c_parse_logfile.py >> data/preprocessing/output/5_SKULLSTRIPPED/parsed_log.txt, you can save
# the decluttered logfile to a new file.
# 
# This script relies on the following file(s) as inputs:
#   * data/preprocessing/output/5_SKULLSTRIPPED/log.txt

#%%
from utils import setup

setup()

lines_to_exclude = [
    'Running SynthStrip from Docker',
    'Configuring model on the CPU',
    'Running SynthStrip model version 1',
    'Input image read from:',
    'Processing frame (of 1):',
    'Masked image saved to:',
    'Binary brain mask saved to:',
    'If you use SynthStrip in your analysis, please cite:',
    '----------------------------------------------------',
    'SynthStrip: Skull-Stripping for Any Brain Image',
    'A Hoopes, JS Mora, AV Dalca, B Fischl, M Hoffmann',
    'NeuroImage 206 (2022), 119474',
    'https://doi.org/10.1016/j.neuroimage.2022.119474',
    'Website: https://w3id.org/synthstrip',
    'Running script 5a_skullstrip.py',
    'Completed 5a_skullstrip.py',
    'Total elapsed time:',
    # 'Using SynthStrip to skull strip'
]

filepath = 'data/preprocessing/output/5a_SKULLSTRIPPED/5a_log.txt'

with open(filepath) as fp:
   line = fp.readline()
   cnt = 1
   while line:
       if not any([line.startswith(s) for s in lines_to_exclude]) and line != '\n':
           print(f"Line {cnt}: {line.strip()}")
       line = fp.readline()
       cnt += 1
# %%
