library(tidyverse)
library(ggplot2)
library(viridis)
library(RColorBrewer)

box_and_whisker <- function(exp) {
  
  exp_title <- str_split(exp, '.csv')[[1]][1]
  
  df <- read_csv(here::here('data', 'radiomics', 'server_results_downloads', exp))
  
  df_long <- df %>%
    mutate(Final_model = factor(Final_model), FeatSelect_model = factor(FeatSelect_model)) %>%
    pivot_longer(cols = c(Val_score, Test_score), names_to = "Score_Type", values_to = "F1_Score") %>% 
    mutate(Score_Type = factor(Score_Type, levels = c("Val_score", "Test_score")))
  
  # Box and whiskers without scatter points
  ggplot(df_long, aes(x = FeatSelect_model, y = F1_Score, fill = Score_Type, color = Num_feats)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.5, position = position_dodge(width = 0.75)) +
    facet_wrap(~Final_model) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_color_viridis_c() + 
    scale_fill_manual(values = c("white", "grey")) +
    ggtitle(paste0(exp_title, ': F1 score broken down by final model'))
  
  ggsave(here::here('data', 'radiomics', 'server_results_downloads', 'box_and_whiskers', paste0(exp_title, '.png')), width = 12, height = 12)
  
  # Box and whiskers with scatter points
  ggplot(df_long, aes(x = FeatSelect_model, y = F1_Score, fill = Score_Type, color = Num_feats)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.5, position = position_dodge(width = 0.75)) +
    geom_point(aes(shape = Score_Type), position = position_dodge(width = 0.75), size = 2) +
    facet_wrap(~Final_model) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_color_viridis_c() + 
    scale_fill_manual(values = c("white", "grey")) +
    ggtitle(paste0(exp_title, ': F1 score broken down by final model'))
  
  ggsave(here::here('data', 'radiomics', 'server_results_downloads', 'box_and_whiskers', paste0('scatter_', exp_title, '.png')), width = 12, height = 12)
}

results_dir <- dir(here::here('data', 'radiomics', 'server_results_downloads'), pattern = "\\.csv$")

results_dir %>% lapply(box_and_whisker)
