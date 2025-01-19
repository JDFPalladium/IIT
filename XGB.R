library(dplyr)
library(caret)
library(xgboost)
library(PRROC)
library(data.table)

setwd("~/February2024")
file.list <- list.files(pattern='iit_prep_0201*')
df.list <- lapply(file.list, readRDS)
iit <- rbindlist(df.list, fill = TRUE)

# Fix Gender - 
iit <- iit %>%
  mutate(Gender = tolower(Gender),
         Gender = ifelse(Gender == "male", "Male", "Female"))

iit <- iit %>%
  mutate(Pregnant = ifelse(!between(Age, 10, 49) | Gender == "Male",
                           "NR", Pregnant),
         Breastfeeding = ifelse(!between(Age, 10, 49) | Gender == "Male",
                                "NR", Breastfeeding))

iit <- iit %>%
  select(-PatientSource, -visit_2, -visit_3, -visit_4, -visit_5, -BMI, -Weight,
         -most_recent_vl, -n_tests_threeyears, -n_hvl_threeyears, -n_lvl_threeyears,
         -recent_hvl_rate, -AHD
         )

gis <- read.csv('../gis_features_dec2023.csv') %>%
  dplyr::select(-Latitude, -Longitude) #%>%
  # mutate(FacilityCode = as.character(FacilityCode))
iit <- merge(iit, gis,by.x = "SiteCode", by.y = "FacilityCode", all.x = TRUE) %>%
  dplyr::select(-SiteCode)

iit$Month <- as.character(iit$Month)
iit$Day <- as.character(iit$Day)

# set.seed(2231)
# split <- createDataPartition(y = iit$Target, p = 0.7, list = FALSE)
# train_data <- iit[split, ]
# test_data <- iit[-split, ]

iit <- iit %>%
  group_by(key) %>%
  mutate(patient_group = sample(1:2, 1, prob = c(0.7, 0.3)))

# helper functions -----------------------------
encodeXGBoost <- function(dataset){
  # Need to one-hot encode all the factor variables
  ohe_features <- names(dataset)[ sapply(dataset, is.factor) | sapply(dataset, is.character) ]
  
  dmy <- dummyVars("~ Month + Gender+  MaritalStatus +  Day+
                   OptimizedHIVRegimen +  Pregnant + DifferentiatedCare + owner_type +
                   StabilityAssessment + most_recent_art_adherence + keph_level_name +
                   Breastfeeding",
                   data = dataset)
  ohe <- data.frame(predict(dmy, newdata = dataset))
  dataset <- cbind(dataset, ohe)
  
  dataset <- dataset %>% select(-ohe_features)
  # dataset[, !(names(dataset) %in% ohe_features)]
  
}

# Let's do temporal cross validation first for sparse -------------------

# train_data <- iit %>% 
#   ungroup() %>%
#   filter(patient_group == 1) %>%
#   select(-patient_group) %>%
#   arrange(PredictionDate) %>%
#   filter(PredictionDate >= "2023-06-01") %>%
#   filter(PredictionDate <= "2023-12-31") 
# 
# test_data <- iit %>% 
#   ungroup() %>%
#   filter(patient_group == 2) %>%
#   select(-patient_group) %>%
#   arrange(PredictionDate) %>%
#   filter(PredictionDate > "2023-01-01") %>%
#   select(-PredictionDate, - key) %>%
#   encodeXGBoost() %>%
#   mutate(PregnantNR = ifelse(GenderFemale == 1 & between(Age, 10, 49), 0, PregnantNR),
#          BreastfeedingNR = ifelse(GenderFemale == 1 & between(Age, 10, 49), 0, BreastfeedingNR))

# # fix pregnant and breastfeeding. if women of child bearing age, then NR = 0
# test_data <- test_data %>%
#   mutate(PregnantNR = ifelse(GenderFemale == 1 & between(Age, 10, 49), 0, PregnantNR),
#          BreastfeedingNR = ifelse(GenderFemale == 1 & between(Age, 10, 49), 0, BreastfeedingNR))


# Let's do a single cross validation
# Create folds
nfolds <- 5
# cuts <- round(seq(1, nrow(train_data), length.out = nfolds+2))
cuts <- seq(min(train_data$PredictionDate), max(train_data$PredictionDate), length.out = nfolds + 2)
# cutgap <- cuts[2]/2

grid_sparse <- expand.grid(model = "xgboost",
                           sparsity = "sparse",
                           eta = c(0.01, 0.1),
                           max_depth = c(6, 8, 10, 12),
                           cs = c(.3, .5, .7),
                           nrounds = c(100, 200, 300, 400, 500))

# set.seed(2231)
# samp <- sample(1:nrow(grid_sparse), 50, replace = FALSE)
# grid_sparse <- grid_sparse[samp, ]

for(i in 1:nrow(grid_sparse)){
  set.seed(2231)
  print(i)
  aucpr <- c()
  
  for(j in 1:nfolds){
    
    # Set train and validation
    tmp <-  train_data %>%
      group_by(key) %>% 
      mutate(patient_group = sample(1:2, 1, prob = c(0.7, 0.3))) 
    
    train_tmp <- tmp %>%
      ungroup() %>%
      filter(patient_group == 1) %>%
      filter(between(PredictionDate, cuts[1], cuts[j+1])) %>%
      select(-PredictionDate, - key, -patient_group) %>%
      encodeXGBoost() %>%
      mutate(PregnantNR = ifelse(GenderFemale == 1 & between(Age, 10, 49), 0, PregnantNR),
             BreastfeedingNR = ifelse(GenderFemale == 1 & between(Age, 10, 49), 0, BreastfeedingNR))
    
    val_tmp <- tmp %>%
      ungroup() %>%
      filter(patient_group == 2) %>%
      filter(between(PredictionDate, cuts[j+1], cuts[j+2])) %>%
      select(-PredictionDate, - key, -patient_group) %>%
      encodeXGBoost() %>%
      mutate(PregnantNR = ifelse(GenderFemale == 1 & between(Age, 10, 49), 0, PregnantNR),
             BreastfeedingNR = ifelse(GenderFemale == 1 & between(Age, 10, 49), 0, BreastfeedingNR))
    
    train_tmp <- train_tmp %>% select(intersect(names(train_tmp), names(val_tmp)))
    val_tmp <- val_tmp %>% select(intersect(names(train_tmp), names(val_tmp)))
    
    dtrain <- xgb.DMatrix(data = data.matrix(train_tmp[,which(names(train_tmp) != "Target")]),
                          label = train_tmp$Target)
    
    set.seed(2231)
    xgb <- xgboost::xgb.train(data = dtrain,
                              eta = grid_sparse[i, 3],
                              max_depth = grid_sparse[i, 4],
                              colsample_bytree = grid_sparse[i, 5],
                              nrounds = grid_sparse[i, 6],
                              objective = "binary:logistic",
                              metric = 'aucpr',
                              verbose = 0
    )
    
    val_predict <- predict(xgb,newdata = data.matrix(val_tmp[, -which(names(val_tmp) == "Target")]))
    fg <- val_predict[val_tmp$Target == 1]
    bg <- val_predict[val_tmp$Target == 0]
    prc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
    aucpr <- c(aucpr, prc$auc.integral)
    print(aucpr)
  }
  print(mean(aucpr))
  grid_sparse$val_pr_auc[i] <- mean(aucpr)
}

saveRDS(grid_sparse, "grid_sparse_slim_02022024.rds")

# save best performing model and thresholds --------------
train_data <- iit %>% 
  ungroup() %>%
  # filter(patient_group == 1) %>% 
  arrange(PredictionDate) %>%
  filter(PredictionDate >= "2023-01-01") %>%
  filter(PredictionDate <= "2023-06-30") %>%
  select(-PredictionDate, - key, -patient_group) %>%
  encodeXGBoost() %>%
  mutate(PregnantNR = ifelse(GenderFemale == 1 & between(Age, 10, 49), 0, PregnantNR),
         BreastfeedingNR = ifelse(GenderFemale == 1 & between(Age, 10, 49), 0, BreastfeedingNR))

dtrain <- xgb.DMatrix(data = data.matrix(train_data[,which(names(train_data) != "Target")]),
                      label = train_data$Target)

set.seed(2231)
xgb <- xgboost::xgb.train(data = dtrain,
                          eta = .01,
                          max_depth = 6,
                          colsample_bytree = 0.3,
                          nrounds = 100,
                          objective = "binary:logistic",
                          verbose = 0
)

train_predict <- predict(xgb,newdata = data.matrix(train_data[, -which(names(train_data) == "Target")]))
fg <- train_predict[train_data$Target == 1]
bg <- train_predict[train_data$Target == 0]
prc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
prc$auc.integral
test_predict <- predict(xgb,newdata = data.matrix(test_data[, -which(names(test_data) == "Target")]))
fg <- test_predict[test_data$Target == 1]
bg <- test_predict[test_data$Target == 0]
prc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
prc$auc.integral

preds_all <- cbind(test_data, test_predict)

# set cutoffs ---------------

all_data <- iit %>%
  ungroup() %>%
  dplyr::select(-patient_group) %>%
  dplyr::select(-PredictionDate, - key) %>%
  encodeXGBoost()


set.seed(2231)
xgb <- xgboost::xgboost(data = data.matrix(all_data %>%select(-Target)),
                        label = all_data$Target,
                        eta = 0.01,
                        max_depth = 8,
                        colsample_bytree = 0.3,
                        nrounds = 500,
                        objective = "binary:logistic",
                        metric = 'aucpr',
                        verbose = 1,
)

val_predict <- predict(xgb,newdata = data.matrix(all_data %>%select(xgb$feature_names)))
val_predict <- cbind(val_predict, all_data)
val_predict <- cbind(val_predict, sitecodes)
# compare by keph level or sumtxcurr


# Set thresholds so that 20% are high risk, 20% are medium risk, 60% are low risk
val_predict <- val_predict %>%
  select(val_predict, Target)

high_thresh <- round(nrow(val_predict)*.2)
medium_thresh <- round(nrow(val_predict)*.4)

val_predict_high <- val_predict %>%
  arrange(desc(val_predict)) %>%
  mutate(rownum = row_number()) %>%
  filter(rownum == high_thresh)
print(val_predict_high$val_predict)

val_predict_medium <- val_predict %>%
  arrange(desc(val_predict)) %>%
  mutate(rownum = row_number()) %>%
  filter(rownum == medium_thresh)
print(val_predict_medium$val_predict)

val_predict_label <- val_predict %>%
  mutate(category = ifelse(val_predict > val_predict_high$val_predict, "High",
                           ifelse(val_predict > val_predict_medium$val_predict, "Medium",
                                  "Low")))

prop.table(table(val_predict_label$category, val_predict_label$Target),
           margin = 2)