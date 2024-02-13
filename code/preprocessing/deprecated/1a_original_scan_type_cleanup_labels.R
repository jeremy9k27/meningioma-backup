# File: 1a_original_scan_type_cleanup_labels.R
# Date: 12/11/23
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description: look at all the different labels present on NURIPs

# package imports
library(tidyverse)
library(readxl)
library(openxlsx)

df <- read_xlsx(here::here("data", "preprocessing", "NURIPS_downloads", "12-11-2023-MeningiomasScanTypeCleanup.xlsx")) %>% 
  mutate(`Series Desc` = factor(`Series Desc`))

lbl_counts <- df %>% 
  group_by(`Series Desc`) %>% 
  summarise(count = n(), min_frames = min(Frames), mean_frames = round(mean(Frames), 2), median_frames = median(Frames), max_frames = max(Frames)) %>% 
  arrange(desc(count), `Series Desc`) 

write_csv(lbl_counts, here::here("data", "preprocessing", "output", "1_scan_type_cleanup", "1a_original_label_counts.csv"))
write.xlsx(lbl_counts, file = here::here("data", "preprocessing", "output", "1_scan_type_cleanup", "1a_original_label_counts.xlsx"))
