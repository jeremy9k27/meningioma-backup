# File: 2b_parse_logfile_warnings.py
# Date: 01/12/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description: Goes through all of the warnings present in 2_NIFTI/log.txt

#--------------------------#
####      CONTENTS      ####
#--------------------------#
# N. Notes
# 1. Set up filepaths
# 2. Parse log file

#--------------------------#
####      N. NOTES      ####
#--------------------------#
# This script is meant to go thru all warnings present in 2_NIFTI/log.txt,
# generating a list of all unique warnings present in the log file. To see all
# warnings, not just those that are most concerning, comment out the warnings in
# the ignore_warnings list below.
#
# This script relies on the following file(s) as inputs:
#   * data/preprocessing/output/2_NIFTI/log.txt

#---------------------------#
#### 1. SET UP FILEPATHS ####
#---------------------------#
from utils import setup

setup()

#-------------------------#
#### 2. PARSE LOG FILE ####
#-------------------------#
# List of warnings to be ignored
ignore_warnings = [
    'Warning: Siemens XA exported as classic not enhanced DICOM (issue 236)',
    'Warning: Siemens MoCo? Bogus slice timing'
]

def should_ignore(warning):
    return any(warning.startswith(ignore_warning) for ignore_warning in ignore_warnings)

def process_file(filename):
    unique_warnings = []
    current_converting = ""

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("Converting"):
                current_converting = line.strip()
            if line.startswith("Warning"):
                warning = line.strip()
                if not should_ignore(warning) and (current_converting, warning) not in unique_warnings:
                    unique_warnings.append((current_converting, warning))

    return unique_warnings

filename = "data/preprocessing/output/2_NIFTI/log.txt"
warnings_found = process_file(filename)
for item in warnings_found:
    print(item)
