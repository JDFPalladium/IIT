library(dplyr)
library(caret)
library(catboost)
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
iit <- readRDS("iit_0314_prepped.rds")

# create time period variable
iit <- iit %>%
  mutate(mt = as.numeric(Month),
         yr = year(apptdate),
         yr = ifelse(yr == "2022", 0, 12),
         tp = mt + yr) %>%
  select(-mt, -yr)

iit$Month <- as.character(iit$Month)
iit$Day <- as.character(iit$Day)

set.seed(2231)
iit <- iit %>% 
  group_by(key) %>% 
  mutate(patient_group = sample(1:2, 1, prob = c(0.8, 0.2)))

train_data <- iit %>% 
  ungroup() %>%
  filter(patient_group == 1) %>%
  select(-patient_group) %>%
  arrange(apptdate) %>%
  filter(apptdate >= "2022-01-01") %>%
  filter(apptdate <= "2023-04-30") 

test_data <- iit %>% 
  ungroup() %>%
  filter(patient_group == 2) %>%
  select(-patient_group) %>%
  arrange(apptdate) %>%
  filter(apptdate >= "2023-05-01") %>%
  filter(apptdate <= "2023-11-30") %>%
  select(-apptdate, - key) 

# Let's do cross validation first for sparse -------------------

# Create folds
nfolds <- 10
cuts <- seq(min(train_data$apptdate), max(train_data$apptdate), length.out = nfolds + 2)

grid_sparse <- expand.grid(model = "catboost",
                           sparsity = "sparse",
                           eta = c(0.01, 0.03, 0.1),
                           max_depth = c(4, 6, 8),
                           nrounds = c(500, 1000, 1500)) %>%
  filter((eta == 0.01 & nrounds == 1500) | (eta == 0.03 & nrounds == 1000) | (eta == 0.1 & nrounds == 500))

for(i in 1:nrow(grid_sparse)){
  set.seed(2231)
  print(i)
  aucpr <- c()
  
  for (j in 1:nfolds) {
    print(j)

    tmp <-  train_data %>%
      group_by(key) %>% 
      mutate(patient_group = sample(1:2, 1, prob = c(0.7, 0.3))) 
    
    train_tmp <- tmp %>%
      ungroup() %>%
      filter(patient_group == 1) %>%
      filter(between(apptdate, cuts[1], cuts[j+1])) %>%
      select(-apptdate, - key, -patient_group) %>%
      as.data.frame()
    
    val_tmp <- tmp %>%
      ungroup() %>%
      filter(patient_group == 2) %>%
      filter(between(apptdate, cuts[j+1], cuts[j+2])) %>%
      select(-apptdate, - key, -patient_group) %>%
      as.data.frame()
    
    train_tmp <- train_tmp %>% select(intersect(names(train_tmp), names(val_tmp))) %>% as.data.frame()
    val_tmp <- val_tmp %>% select(intersect(names(train_tmp), names(val_tmp))) %>% as.data.frame()
    
    char_cols <- sapply(train_tmp, is.character)
    train_tmp[char_cols] <- lapply(train_tmp[char_cols], as.factor)
    
    # Create a CatBoost training dataset
    train_tmp_cb <- catboost.load_pool(data = train_tmp[, which(names(train_tmp)!="Target")],
                                     label = train_tmp$Target)
    
    val_tmp[char_cols] <- lapply(val_tmp[char_cols], as.factor)
    
    # Create a CatBoost training dataset
    val_tmp_cb <- catboost.load_pool(data = val_tmp[, which(names(val_tmp)!="Target")],
                                       label = val_tmp$Target)
    
    # Set up CatBoost parameters
    params <- list(
      iterations = grid_sparse[i, 5],  # Number of boosting iterations
      learning_rate = grid_sparse[i, 3],  # Learning rate
      depth = grid_sparse[i, 4]  # Depth of the trees
      # Add more parameters as needed
    )
    
    # Train the CatBoost model
    model <- catboost.train(
      train_tmp_cb,
      params = params
    )
    
    val_predict <- catboost.predict(model, 
                                   val_tmp_cb, 
                                   prediction_type = 'RawFormulaVal')
    fg <- val_predict[val_tmp$Target == 1]
    bg <- val_predict[val_tmp$Target == 0]
    prc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
    aucpr <- c(aucpr, prc$auc.integral)
    # print(aucpr)
  }
  print(mean(aucpr))
  grid_sparse$val_pr_auc[i] <- mean(aucpr)
}

saveRDS(grid_sparse, "Results/catboost_sparse_tempgroup_03182024.rds")

train_final <- train_data %>%
  ungroup() %>%
  select(-apptdate, - key) %>%
  as.data.frame()

char_cols <- sapply(train_final, is.character)
train_final[char_cols] <- lapply(train_final[char_cols], as.factor)

# Create a CatBoost training dataset
train_tmp_cb <- catboost.load_pool(data = train_final[, which(names(train_final)!="Target")],
                                   label = train_final$Target)

test_data[char_cols] <- lapply(test_data[char_cols], as.factor)

# Create a CatBoost training dataset
test_tmp_cb <- catboost.load_pool(data = test_data[, which(names(test_data)!="Target")],
                                 label = test_data$Target)

# Set up CatBoost parameters
params <- list(
  iterations = 1000,  # Number of boosting iterations
  learning_rate = 0.03,  # Learning rate
  depth = 8  # Depth of the trees
  # Add more parameters as needed
)

# Train the CatBoost model
model <- catboost.train(
  train_tmp_cb,
  params = params
)

val_predict <- catboost.predict(model, 
                                test_tmp_cb, 
                                prediction_type = 'RawFormulaVal')
fg <- val_predict[test_data$Target == 1]
bg <- val_predict[test_data$Target == 0]
prc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
prc # .109
roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
roc # .783

# Sparse rebalanced ----------------------------

# Create folds
nfolds <- 10
cuts <- seq(min(train_data$apptdate), max(train_data$apptdate), length.out = nfolds + 2)

# Count the occurrences of each class in the Target variable
class_counts <- table(train_data$Target)

# Calculate the desired number of samples for each class
desired_negatives <- round(min(class_counts) * 10 / 4.2)

# Subset the dataframe to include only the minority class
minority_class <- train_data[train_data$Target == 1, ]

# Subset the dataframe to include only the majority class
majority_class <- train_data[train_data$Target == 0, ]

# Sample the majority class to achieve desired_samples
set.seed(2231)
sampled_majority <- majority_class[sample(nrow(majority_class), desired_negatives), ]

# Combine the minority class and the sampled majority class
balanced_data <- rbind(minority_class, sampled_majority)

grid_sparse <- expand.grid(model = "catboost",
                           sparsity = "sparse_bal",
                           eta = c(0.01, 0.03, 0.1),
                           max_depth = c(4, 6, 8),
                           nrounds = c(500, 1000, 1500)) %>%
  filter((eta == 0.01 & nrounds == 1500) | (eta == 0.03 & nrounds == 1000) | (eta == 0.1 & nrounds == 500))

for(i in 1:nrow(grid_sparse)){
  set.seed(2231)
  print(i)
  aucpr <- c()
  
  for (j in 1:nfolds) {
    print(j)
    
    tmp <-  balanced_data %>%
      group_by(key) %>% 
      mutate(patient_group = sample(1:2, 1, prob = c(0.7, 0.3))) 
    
    train_tmp <- tmp %>%
      ungroup() %>%
      filter(patient_group == 1) %>%
      filter(between(apptdate, cuts[1], cuts[j+1])) %>%
      select(-apptdate, - key, -patient_group) %>%
      as.data.frame()
    
    val_tmp <- tmp %>%
      ungroup() %>%
      filter(patient_group == 2) %>%
      filter(between(apptdate, cuts[j+1], cuts[j+2])) %>%
      select(-apptdate, - key, -patient_group) %>%
      as.data.frame()
    
    train_tmp <- train_tmp %>% select(intersect(names(train_tmp), names(val_tmp))) %>% as.data.frame()
    val_tmp <- val_tmp %>% select(intersect(names(train_tmp), names(val_tmp))) %>% as.data.frame()
    
    char_cols <- sapply(train_tmp, is.character)
    train_tmp[char_cols] <- lapply(train_tmp[char_cols], as.factor)
    
    # Create a CatBoost training dataset
    train_tmp_cb <- catboost.load_pool(data = train_tmp[, which(names(train_tmp)!="Target")],
                                       label = train_tmp$Target)
    
    val_tmp[char_cols] <- lapply(val_tmp[char_cols], as.factor)
    
    # Create a CatBoost training dataset
    val_tmp_cb <- catboost.load_pool(data = val_tmp[, which(names(val_tmp)!="Target")],
                                     label = val_tmp$Target)
    
    # Set up CatBoost parameters
    params <- list(
      iterations = grid_sparse[i, 5],  # Number of boosting iterations
      learning_rate = grid_sparse[i, 3],  # Learning rate
      depth = grid_sparse[i, 4]  # Depth of the trees
      # Add more parameters as needed
    )
    
    # Train the CatBoost model
    model <- catboost.train(
      train_tmp_cb,
      params = params
    )
    
    val_predict <- catboost.predict(model, 
                                    val_tmp_cb, 
                                    prediction_type = 'RawFormulaVal')
    fg <- val_predict[val_tmp$Target == 1]
    bg <- val_predict[val_tmp$Target == 0]
    prc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
    aucpr <- c(aucpr, prc$auc.integral)
    # print(aucpr)
  }
  print(mean(aucpr))
  grid_sparse$val_pr_auc[i] <- mean(aucpr)
}

saveRDS(grid_sparse, "Results/catboost_sparse_tempgroupbal_03182024.rds")

train_final <- balanced_data %>%
  ungroup() %>%
  select(-apptdate, - key) %>%
  as.data.frame()

char_cols <- sapply(train_final, is.character)
train_final[char_cols] <- lapply(train_final[char_cols], as.factor)

# Create a CatBoost training dataset
train_tmp_cb <- catboost.load_pool(data = train_final[, which(names(train_final)!="Target")],
                                   label = train_final$Target)

test_data[char_cols] <- lapply(test_data[char_cols], as.factor)

# Create a CatBoost training dataset
test_tmp_cb <- catboost.load_pool(data = test_data[, which(names(test_data)!="Target")],
                                  label = test_data$Target)

# Set up CatBoost parameters
params <- list(
  iterations = 1000,  # Number of boosting iterations
  learning_rate = 0.03,  # Learning rate
  depth = 8  # Depth of the trees
    # Add more parameters as needed
)

# Train the CatBoost model
model <- catboost.train(
  train_tmp_cb,
  params = params
)

val_predict <- catboost.predict(model, 
                                test_tmp_cb, 
                                prediction_type = 'RawFormulaVal')
fg <- val_predict[test_data$Target == 1]
bg <- val_predict[test_data$Target == 0]
prc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
prc #.1049
roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
roc # .788




# MICE --------------------------
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

grid_mice <- expand.grid(model = "catboost",
                          sparsity = "mice",
                         eta = c(0.01, 0.03, 0.1),
                         max_depth = c(4, 6, 8),
                         nrounds = c(500, 1000, 1500)) %>%
  filter((eta == 0.01 & nrounds == 1500) | (eta == 0.03 & nrounds == 1000) | (eta == 0.1 & nrounds == 500))

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
      select(-Cat) 
    
    val_tmp <- tmp %>%
      ungroup() %>%
      filter(Cat == "val") %>%
      select(-Cat) 
    
    train_tmp <- train_tmp %>% select(intersect(names(train_tmp), names(val_tmp))) %>% as.data.frame()
    val_tmp <- val_tmp %>% select(intersect(names(train_tmp), names(val_tmp))) %>% as.data.frame()
    
    char_cols <- sapply(train_tmp, is.character)
    train_tmp[char_cols] <- lapply(train_tmp[char_cols], as.factor)
    
    # Create a CatBoost training dataset
    train_tmp_cb <- catboost.load_pool(data = train_tmp[, which(names(train_tmp)!="Target")],
                                       label = train_tmp$Target)
    
    val_tmp[char_cols] <- lapply(val_tmp[char_cols], as.factor)
    
    # Create a CatBoost training dataset
    val_tmp_cb <- catboost.load_pool(data = val_tmp[, which(names(val_tmp)!="Target")],
                                     label = val_tmp$Target)
    
    # Set up CatBoost parameters
    params <- list(
      iterations = grid_mice[i, 5],  # Number of boosting iterations
      learning_rate = grid_mice[i, 3],  # Learning rate
      depth = grid_mice[i, 4]  # Depth of the trees
      # Add more parameters as needed
    )
    
    # Train the CatBoost model
    model <- catboost.train(
      train_tmp_cb,
      params = params
    )
    
    val_predict <- catboost.predict(model, 
                                    val_tmp_cb, 
                                    prediction_type = 'RawFormulaVal')
    fg <- val_predict[val_tmp$Target == 1]
    bg <- val_predict[val_tmp$Target == 0]
    prc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
    aucpr <- c(aucpr, prc$auc.integral)
    print(aucpr)
  }
  print(mean(aucpr))
  grid_mice$val_pr_auc[i] <- mean(aucpr)
}

saveRDS(grid_mice, "Results/catboost_mice_03182024.rds")

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

char_cols <- sapply(train_final, is.character)
train_final[char_cols] <- lapply(train_final[char_cols], as.factor)

# Create a CatBoost training dataset
train_tmp_cb <- catboost.load_pool(data = train_final[, which(names(train_final)!="Target")],
                                   label = train_final$Target)

test_data[char_cols] <- lapply(test_data[char_cols], as.factor)

# Create a CatBoost training dataset
test_tmp_cb <- catboost.load_pool(data = test_data[, which(names(test_data)!="Target")],
                                  label = test_data$Target)

# Set up CatBoost parameters
params <- list(
  iterations = 1000,  # Number of boosting iterations
  learning_rate = 0.03,  # Learning rate
  depth = 8  # Depth of the trees
  # Add more parameters as needed
)

# Train the CatBoost model
model <- catboost.train(
  train_tmp_cb,
  params = params
)

val_predict <- catboost.predict(model, 
                                test_tmp_cb, 
                                prediction_type = 'RawFormulaVal')
fg <- val_predict[test_data$Target == 1]
bg <- val_predict[test_data$Target == 0]
prc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
prc # .094
roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
roc # .777


# MICE --------------------------
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

grid_mice <- expand.grid(model = "catboost",
                         sparsity = "mice_bal",
                         eta = c(0.01, 0.03, 0.1),
                         max_depth = c(4, 6, 8),
                         nrounds = c(500, 1000, 1500)) %>%
  filter((eta == 0.01 & nrounds == 1500) | (eta == 0.03 & nrounds == 1000) | (eta == 0.1 & nrounds == 500))

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
      select(-Cat) 
    
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
      select(-Cat) 
    
    train_tmp <- train_tmp %>% select(intersect(names(train_tmp), names(val_tmp))) %>% as.data.frame()
    val_tmp <- val_tmp %>% select(intersect(names(train_tmp), names(val_tmp))) %>% as.data.frame()
    
    char_cols <- sapply(train_tmp, is.character)
    train_tmp[char_cols] <- lapply(train_tmp[char_cols], as.factor)
    
    # Create a CatBoost training dataset
    train_tmp_cb <- catboost.load_pool(data = train_tmp[, which(names(train_tmp)!="Target")],
                                       label = train_tmp$Target)
    
    val_tmp[char_cols] <- lapply(val_tmp[char_cols], as.factor)
    
    # Create a CatBoost training dataset
    val_tmp_cb <- catboost.load_pool(data = val_tmp[, which(names(val_tmp)!="Target")],
                                     label = val_tmp$Target)
    
    # Set up CatBoost parameters
    params <- list(
      iterations = grid_mice[i, 5],  # Number of boosting iterations
      learning_rate = grid_mice[i, 3],  # Learning rate
      depth = grid_mice[i, 4]  # Depth of the trees
      # Add more parameters as needed
    )
    
    # Train the CatBoost model
    model <- catboost.train(
      train_tmp_cb,
      params = params
    )
    
    val_predict <- catboost.predict(model, 
                                    val_tmp_cb, 
                                    prediction_type = 'RawFormulaVal')
    fg <- val_predict[val_tmp$Target == 1]
    bg <- val_predict[val_tmp$Target == 0]
    prc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
    aucpr <- c(aucpr, prc$auc.integral)
    print(aucpr)
  }
  print(mean(aucpr))
  grid_mice$val_pr_auc[i] <- mean(aucpr)
}

saveRDS(grid_mice, "Results/catboost_mice_bal_03182024.rds")

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

test_data <- mice_full %>%
  filter(Cat == "val") %>%
  select(-Cat) %>%
  as.data.frame() %>%
  mutate(tp = as.numeric(Month) + 12,
         Month = as.character(Month),
         Day = as.character(Day))

char_cols <- sapply(train_final, is.character)
train_final[char_cols] <- lapply(train_final[char_cols], as.factor)

# Create a CatBoost training dataset
train_tmp_cb <- catboost.load_pool(data = train_final[, which(names(train_final)!="Target")],
                                   label = train_final$Target)

test_data[char_cols] <- lapply(test_data[char_cols], as.factor)

# Create a CatBoost training dataset
test_tmp_cb <- catboost.load_pool(data = test_data[, which(names(test_data)!="Target")],
                                  label = test_data$Target)

# Set up CatBoost parameters
params <- list(
  iterations = 1500,  # Number of boosting iterations
  learning_rate = 0.01,  # Learning rate
  depth = 8  # Depth of the trees
  # Add more parameters as needed
)

# Train the CatBoost model
model <- catboost.train(
  train_tmp_cb,
  params = params
)

val_predict <- catboost.predict(model, 
                                test_tmp_cb, 
                                prediction_type = 'RawFormulaVal')
fg <- val_predict[test_data$Target == 1]
bg <- val_predict[test_data$Target == 0]
prc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
prc # .0997
roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
roc # .7789
