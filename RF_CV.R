library(dplyr)
library(caret)
library(ranger)
library(PRROC)
library(data.table)
library(aws.s3)
Sys.setenv("AWS_ACCESS_KEY_ID" = "",
           "AWS_SECRET_ACCESS_KEY" = "",
           "AWS_DEFAULT_REGION" = "")

# name the bucket
aws_bucket <- ""

# get the bucket
get_bucket(aws_bucket)

setwd("~/February2024")

# # Simple ---------------------
# 
# grid_simple <- expand.grid(model = "rf",
#                            sparsity = "simple",
#                            mtry = c(5,7,9),
#                            min.node.size = c(10, 5))
# 
# Mode <- function(x) {
#   ux <- unique(x)
#   ux[which.max(tabulate(match(x, ux)))]
# }
# replaceWithMode <- function(dataset_calc, dataset_impute, position){
#   dataset_impute[, position] <- ifelse(is.na(dataset_impute[, position]),
#                                        Mode(dataset_calc[, position][!is.na(dataset_calc[,position])]),
#                                        dataset_impute[, position])
# }
# replaceWithMean <- function(dataset_calc, dataset_impute, position){
#   dataset_impute[, position] <- ifelse(is.na(dataset_impute[, position]),
#                                        mean(dataset_calc[, position], na.rm = TRUE),
#                                        dataset_impute[, position])
# }
# 
# train_data <- iit[split, ]
# test_data <- iit[-split, ]
# 
# train_data <- train_data %>% select(-key, -PredictionDate)
# 
# for(i in 1:nrow(grid_simple)){
#   set.seed(2231)
#   print(i)
#   aucpr <- c()
#   
#   for (j in seq_along(folds)) {
#     print(j)
#     train_indices <- unlist(folds[-j])
#     val_indices <- unlist(folds[j])
#     
#     train_tmp <- train_data[train_indices, ]
#     val_tmp <- train_data[val_indices, ]
#     
#     train_tmp <- as.data.frame(train_tmp)
#     val_tmp <- as.data.frame(val_tmp)
#     
#     for(k in which(sapply(train_tmp, class) %in% c('character', 'factor'))){
#       train_tmp[, k] <- replaceWithMode(train_tmp, train_tmp, k)
#       val_tmp[, k] <- replaceWithMode(train_tmp, val_tmp, k)
#     }
#     
#     for(k in which(sapply(train_tmp, class) %in% c('numeric', 'integer'))){
#       train_tmp[, k] <- replaceWithMean(train_tmp, train_tmp, k)
#       val_tmp[, k] <- replaceWithMean(train_tmp, val_tmp, k)
#     }
# 
#     set.seed(2231)
#     rf <- ranger(
#       dependent.variable.name = 'Target',
#       data = train_tmp,
#       mtry = grid_simple[i, 3],
#       min.node.size = grid_simple[i, 4],
#       num.threads = 64
#     )
#     
#     pred_val = predict(rf, data=val_tmp)
#     fg <- pred_val$predictions[val_tmp$Target == 1]
#     bg <- pred_val$predictions[val_tmp$Target == 0]
#     prc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
#     aucpr <- c(aucpr, prc$auc.integral)
#     # print(aucpr)
#   }
#   print(mean(aucpr))
#   grid_simple$val_pr_auc[i] <- mean(aucpr)
# }
# 
# saveRDS(grid_simple, "Results/rf_grid_simple_regcv_02042024.rds")

# Mice --------------------

setwd("~/February2024")
iit <- readRDS("iit_0314_prepped.rds")

train_data <- iit %>% 
  ungroup() %>%
  arrange(apptdate) %>%
  filter(apptdate >= "2022-01-01") %>%
  filter(apptdate <= "2023-04-30") 

fold1 <- readRDS("iit_mice_03142024_1.rds")
fold2 <- readRDS("iit_mice_03142024_2.rds")
fold3 <- readRDS("iit_mice_03142024_3.rds")
fold4 <- readRDS("iit_mice_03142024_4.rds")
fold5 <- readRDS("iit_mice_03142024_5.rds")
fold6 <- readRDS("iit_mice_03142024_6.rds")
fold7 <- readRDS("iit_mice_03142024_7.rds")
fold8 <- readRDS("iit_mice_03142024_8.rds")
fold9 <- readRDS("iit_mice_03142024_9.rds")
fold10 <- readRDS("iit_mice_03142024_10.rds")
folds <- list(fold1, fold2, fold3, fold4, fold5,
              fold6, fold7, fold8, fold9, fold10)

# Create folds
nfolds <- 10
cuts <- seq(min(train_data$apptdate), max(train_data$apptdate), length.out = nfolds)

grid_mice <- expand.grid(model = "rf",
                           sparsity = "mice",
                           mtry = c(5,7,9),
                           min.node.size = c(10, 5))

for(i in 1:nrow(grid_mice)){
  set.seed(2231)
  print(i)
  aucpr <- c()
  
  for (j in 1:nfolds) {
    print(j)
    
    tmp <- folds[[j]]
    
    tmp$tp <- as.numeric(tmp$Month)
    
    tmp$Month <- as.character(tmp$Month)
    tmp$Day <- as.character(tmp$Day)
    
    train_tmp <- tmp %>%
      ungroup() %>%
      filter(Cat == "train") %>%
      select(-Cat) %>%
      as.data.frame()
    
    val_tmp <- tmp %>%
      ungroup() %>%
      filter(Cat == "val") %>%
      select(-Cat) %>%
      as.data.frame()
    
    set.seed(2231)
    rf <- ranger(
      dependent.variable.name = 'Target',
      data = train_tmp,
      mtry = grid_mice[i, 3],
      min.node.size = grid_mice[i, 4],
      num.threads = 64
    )
    
    pred_val = predict(rf, data=val_tmp)
    fg <- pred_val$predictions[val_tmp$Target == 1]
    bg <- pred_val$predictions[val_tmp$Target == 0]
    prc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
    aucpr <- c(aucpr, prc$auc.integral)
    # print(aucpr)
  }
  print(mean(aucpr))
  grid_mice$val_pr_auc[i] <- mean(aucpr)
}

saveRDS(grid_mice, "Results/rf_grid_mice_03182024.rds")

# load in complete dataset
mice_full <- readRDS("iit_mice_03142024_full.rds")

train_final <- mice_full %>%
  filter(Cat == "train") %>%
  select(-Cat) %>%
  as.data.frame() %>%
  mutate(tp = as.numeric(Month),
         Month = as.character(Month),
         Day = as.character(Day))

test_data <- mice_full %>%
  filter(Cat == "val") %>%
  select(-Cat) %>%
  as.data.frame() %>%
  mutate(tp = as.numeric(Month) + 12,
         Month = as.character(Month),
         Day = as.character(Day))

set.seed(2231)
rf <- ranger(
  dependent.variable.name = 'Target',
  data = train_final,
  mtry = 5,
  min.node.size = 10,
  num.threads = 64
)

pred_val = predict(rf, data=test_data)
fg <- pred_val$predictions[test_data$Target == 1]
bg <- pred_val$predictions[test_data$Target == 0]
prc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
prc # .094
roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
roc # 0.764

# Rebalance ------------------

setwd("~/February2024")
iit <- readRDS("iit_0314_prepped.rds")

train_data <- iit %>% 
  ungroup() %>%
  arrange(apptdate) %>%
  filter(apptdate >= "2022-01-01") %>%
  filter(apptdate <= "2023-04-30") 

fold1 <- readRDS("iit_mice_03142024_1.rds")
fold2 <- readRDS("iit_mice_03142024_2.rds")
fold3 <- readRDS("iit_mice_03142024_3.rds")
fold4 <- readRDS("iit_mice_03142024_4.rds")
fold5 <- readRDS("iit_mice_03142024_5.rds")
fold6 <- readRDS("iit_mice_03142024_6.rds")
fold7 <- readRDS("iit_mice_03142024_7.rds")
fold8 <- readRDS("iit_mice_03142024_8.rds")
fold9 <- readRDS("iit_mice_03142024_9.rds")
fold10 <- readRDS("iit_mice_03142024_10.rds")
folds <- list(fold1, fold2, fold3, fold4, fold5,
              fold6, fold7, fold8, fold9, fold10)

# Create folds
nfolds <- 10
cuts <- seq(min(train_data$apptdate), max(train_data$apptdate), length.out = nfolds)

grid_mice <- expand.grid(model = "rf",
                         sparsity = "mice_bal",
                         mtry = c(5,7,9),
                         min.node.size = c(10, 5))

for(i in 1:nrow(grid_mice)){
  set.seed(2231)
  print(i)
  aucpr <- c()
  
  for (j in 1:nfolds) {
    print(j)
    
    tmp <- folds[[j]]
    
    tmp$tp <- as.numeric(tmp$Month)
    
    tmp$Month <- as.character(tmp$Month)
    tmp$Day <- as.character(tmp$Day)
    
    train_tmp <- tmp %>%
      ungroup() %>%
      filter(Cat == "train") %>%
      select(-Cat) %>%
      as.data.frame()
    
    # Count the occurrences of each class in the Target variable
    class_counts <- table(train_tmp$Target)
  
    # Calculate the desired number of samples for each class
    desired_negatives <- round(min(class_counts) * 10 / 4.2)
    
    # Subset the dataframe to include only the minority class
    minority_class <- train_tmp[train_tmp$Target == 1, ]
    
    # Subset the dataframe to include only the majority class
    majority_class <- train_tmp[train_tmp$Target == 0, ]
    
    # Sample the majority class to achieve desired_samples
    set.seed(2231)
    sampled_majority <- majority_class[sample(nrow(majority_class), desired_negatives), ]
    
    # Combine the minority class and the sampled majority class
    train_tmp <- rbind(minority_class, sampled_majority)
    
    val_tmp <- tmp %>%
      ungroup() %>%
      filter(Cat == "val") %>%
      select(-Cat) %>%
      as.data.frame()
    
    set.seed(2231)
    rf <- ranger(
      dependent.variable.name = 'Target',
      data = train_tmp,
      mtry = grid_mice[i, 3],
      min.node.size = grid_mice[i, 4],
      num.threads = 64
    )
    
    pred_val = predict(rf, data=val_tmp)
    fg <- pred_val$predictions[val_tmp$Target == 1]
    bg <- pred_val$predictions[val_tmp$Target == 0]
    prc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
    aucpr <- c(aucpr, prc$auc.integral)
    # print(aucpr)
  }
  print(mean(aucpr))
  grid_mice$val_pr_auc[i] <- mean(aucpr)
}

saveRDS(grid_mice, "Results/rf_grid_micebal_03182024.rds")

# load in complete dataset
mice_full <- readRDS("iit_mice_03142024_full.rds")

train_final <- mice_full %>%
  filter(Cat == "train") %>%
  select(-Cat) %>%
  as.data.frame() %>%
  mutate(tp = as.numeric(Month),
         Month = as.character(Month),
         Day = as.character(Day))

# Count the occurrences of each class in the Target variable
class_counts <- table(train_final$Target)

# Calculate the desired number of samples for each class
desired_negatives <- round(min(class_counts) * 10 / 4.2)

# Subset the dataframe to include only the minority class
minority_class <- train_final[train_final$Target == 1, ]

# Subset the dataframe to include only the majority class
majority_class <- train_final[train_final$Target == 0, ]

# Sample the majority class to achieve desired_samples
set.seed(2231)
sampled_majority <- majority_class[sample(nrow(majority_class), desired_negatives), ]

# Combine the minority class and the sampled majority class
train_final <- rbind(minority_class, sampled_majority)

test_data <- mice_full %>%
  filter(Cat == "val") %>%
  select(-Cat) %>%
  as.data.frame() %>%
  mutate(tp = as.numeric(Month) + 12,
         Month = as.character(Month),
         Day = as.character(Day))

set.seed(2231)
rf <- ranger(
  dependent.variable.name = 'Target',
  data = train_final,
  mtry = 9,
  min.node.size = 5,
  num.threads = 64
)

pred_val = predict(rf, data=test_data)
fg <- pred_val$predictions[test_data$Target == 1]
bg <- pred_val$predictions[test_data$Target == 0]
prc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
prc # .0957
roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
roc # 0.7706



