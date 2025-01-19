library(dplyr)
library(caret)
library(xgboost)
library(PRROC)
library(data.table)
library(SHAPforxgboost)

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
  
  # dataset <- dataset %>% select(all_of(-ohe_features))
  dataset[, !(names(dataset) %in% ohe_features)]
  
}

encodeXGBoostNoLocVars <- function(dataset){
  # Need to one-hot encode all the factor variables
  ohe_features <- names(dataset)[ sapply(dataset, is.factor) | sapply(dataset, is.character) ]
  
  dmy <- dummyVars("~ Month + Gender+  MaritalStatus +  Day+
                   OptimizedHIVRegimen +  Pregnant + DifferentiatedCare +
                   StabilityAssessment + most_recent_art_adherence + Breastfeeding",
                   data = dataset)
  ohe <- data.frame(predict(dmy, newdata = dataset))
  dataset <- cbind(dataset, ohe)
  
  # dataset <- dataset %>% select(all_of(-ohe_features))
  dataset[, !(names(dataset) %in% ohe_features)]
  
}

# overall results ------------------------

train_data <- iit %>% 
  ungroup() %>%
  filter(patient_group == 1) %>%
  select(-patient_group) %>%
  arrange(apptdate) %>%
  filter(apptdate >= "2022-01-01") %>%
  filter(apptdate <= "2023-04-30") 

# Rebalance to 70-30
# Assuming your dataset is stored in a dataframe called "data" and the outcome variable is called "Target"

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


test_data <- iit %>% 
  ungroup() %>%
  filter(patient_group == 2) %>%
  select(-patient_group) %>%
  arrange(apptdate) %>%
  filter(apptdate >= "2023-05-01") %>%
  filter(apptdate <= "2023-11-30") %>%
  select(-apptdate, - key) %>%
  encodeXGBoost() %>%
  mutate(PregnantNR = ifelse(GenderFemale == 1 & between(Age, 10, 49), 0, PregnantNR),
         BreastfeedingNR = ifelse(GenderFemale == 1 & between(Age, 10, 49), 0, BreastfeedingNR)) %>%
  mutate(MonthDec = 0,
         MonthJan = 0,
         MonthFeb = 0,
         MonthMar = 0,
         MonthApr = 0) %>%
  as.data.frame()


# train_data or balanced_data
train_final <- balanced_data %>%
  select(-apptdate, -key) %>%
  encodeXGBoost() %>%
  mutate(PregnantNR = ifelse(GenderFemale == 1 & between(Age, 10, 49), 0, PregnantNR),
         BreastfeedingNR = ifelse(GenderFemale == 1 & between(Age, 10, 49), 0, BreastfeedingNR)) %>%
  as.data.frame()

dtrain <- xgb.DMatrix(data = data.matrix(train_final[,which(names(train_final) != "Target")]),
                      label = train_final$Target)

set.seed(2231)
xgb <- xgboost::xgb.train(data = dtrain,
                          eta = 0.01,
                          max_depth = 8,
                          colsample_bytree = 0.5,
                          nrounds = 500,
                          objective = "binary:logistic",
                          # metric = 'aucpr',
                          verbose = 0
)

td <- test_data %>% select(names(train_final)) %>% select(-Target)
test_predict <- predict(xgb,newdata = data.matrix(td))
fg <- test_predict[test_data$Target == 1]
bg <- test_predict[test_data$Target == 0]
prc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
prc$auc.integral #0.11058
roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
roc$auc #.7859


# how did it perform for children vs adults and men vs women -----------------
children <- test_data %>% select(names(train_final)) %>% filter(Age < 15) # 1.4% IIT
test_predict <- predict(xgb,newdata = data.matrix(children %>% select(-Target)))
fg <- test_predict[children$Target == 1]
bg <- test_predict[children$Target == 0]
prc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
prc$auc.integral #0.075
roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
roc$auc #.766

adults <- test_data %>% select(names(train_final)) %>% filter(Age >= 15)
prop.table(table(adults$Target)) # 2.1%
test_predict <- predict(xgb,newdata = data.matrix(adults %>% select(-Target)))
fg <- test_predict[adults$Target == 1]
bg <- test_predict[adults$Target == 0]
prc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
prc$auc.integral #0.1118
roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
roc$auc #.786

# how did it perform for men vs women
men <- test_data %>% select(names(train_final)) %>% filter(GenderMale == 1 & Age >= 15)
prop.table(table(men$Target)) # 2.57%
test_predict <- predict(xgb,newdata = data.matrix(men %>% select(-Target)))
fg <- test_predict[men$Target == 1]
bg <- test_predict[men$Target == 0]
prc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
prc$auc.integral #0.1244
roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
roc$auc #.7911

women <- test_data %>% select(names(train_final)) %>% filter(GenderMale == 0 & Age >= 15) 
prop.table(table(women$Target)) # 1.88%
test_predict <- predict(xgb,newdata = data.matrix(women %>% select(-Target)))
fg <- test_predict[women$Target == 1]
bg <- test_predict[women$Target == 0]
prc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
prc$auc.integral #0.1039
roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
roc$auc #.7809

# Let's try without locational variables --------------
loc_vars <- c("births", "pregnancies", "literacy", "poverty", "anc", "pnc", "sba", "hiv_prev", "hiv_count", "condom",
              "intercourse", "in_union", "circumcision", "partner_away", "partner_men", "partner_women", "sti", "pop",
              "keph_level_name", "owner_type", "SumTXCurr")

train_data <- iit %>% 
  ungroup() %>%
  filter(patient_group == 1) %>%
  select(-patient_group) %>%
  select(-loc_vars) %>%
  arrange(apptdate) %>%
  filter(apptdate >= "2022-01-01") %>%
  filter(apptdate <= "2023-04-30") 

# Rebalance to 70-30
# Assuming your dataset is stored in a dataframe called "data" and the outcome variable is called "Target"

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


test_data <- iit %>% 
  ungroup() %>%
  filter(patient_group == 2) %>%
  select(-patient_group) %>%
  select(-loc_vars) %>%
  arrange(apptdate) %>%
  filter(apptdate >= "2023-05-01") %>%
  filter(apptdate <= "2023-11-30") %>%
  select(-apptdate, - key) %>%
  encodeXGBoostNoLocVars() %>%
  mutate(PregnantNR = ifelse(GenderFemale == 1 & between(Age, 10, 49), 0, PregnantNR),
         BreastfeedingNR = ifelse(GenderFemale == 1 & between(Age, 10, 49), 0, BreastfeedingNR)) %>%
  mutate(MonthDec = 0,
         MonthJan = 0,
         MonthFeb = 0,
         MonthMar = 0,
         MonthApr = 0) %>%
  as.data.frame()


# train_data or balanced_data
train_final <- balanced_data %>%
  select(-apptdate, -key) %>%
  encodeXGBoostNoLocVars() %>%
  mutate(PregnantNR = ifelse(GenderFemale == 1 & between(Age, 10, 49), 0, PregnantNR),
         BreastfeedingNR = ifelse(GenderFemale == 1 & between(Age, 10, 49), 0, BreastfeedingNR)) %>%
  as.data.frame()

dtrain <- xgb.DMatrix(data = data.matrix(train_final[,which(names(train_final) != "Target")]),
                      label = train_final$Target)

set.seed(2231)
xgb_loc <- xgboost::xgb.train(data = dtrain,
                          eta = 0.01,
                          max_depth = 8,
                          colsample_bytree = 0.5,
                          nrounds = 500,
                          objective = "binary:logistic",
                          # metric = 'aucpr',
                          verbose = 0
)


td <- test_data %>% select(names(train_final)) %>% select(-Target)
test_predict <- predict(xgb_loc,newdata = data.matrix(td))
fg <- test_predict[test_data$Target == 1]
bg <- test_predict[test_data$Target == 0]
prc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
prc$auc.integral #0.967
roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
roc$auc #.7708


# SHAP -----------------
shap_values <- shap.prep(xgb_model = xgb,
                         X_train = data.matrix(train_final[,which(names(train_final) != "Target")]))

# Join to get feature category
imp <- read.csv("iit_feature_table.csv")

shap_values1 <- merge(shap_values, imp,
                      by.x = "variable", 
                      by.y = "Feature")

# get mean values
shap_unique <- unique(shap_values1[, c("Feature_Category", "mean_value")])
shap_summary <- shap_unique %>%
  group_by(Feature_Category) %>%
  summarize(mean_value = sum(mean_value)) 
# rename variable to feature category
# group by variable and get new mean_value, stdfvalue
shap_values2 <- shap_values1 %>%
  select(Feature_Category, value, rfvalue, stdfvalue) %>%
  merge(., shap_summary, by = "Feature_Category") %>%
  rename(variable = Feature_Category) 
saveRDS(shap_values2, "shap_values2.rds")
shap_values2 <- readRDS("shap_values2.rds") 
## Plot shap overall metrics
# Get 25 top features
df <- unique(shap_values2[, c("variable", "mean_value")]) %>%
  arrange(desc(mean_value)) 
# Barplot
shap_values2 <- shap_values2 %>% filter(variable %in% df$variable[1:15])
shap_values2$variable[shap_values2$variable == "Tp"] <- "TimePeriod"
shap_values2$variable <- factor(shap_values2$variable,
                                levels = c("AverageLatenessLast5",
                                           "DaystoNextAppointment",
                                           "Circumcision",
                                           "AverageLatenessLast3",
                                           "AverageLatenessLast10",
                                           "AverageLateness",
                                           "TimeonART",
                                           "LateRate",
                                           "LateLast1",
                                           "Age",
                                           "UnscheduledRate",
                                           "AverageTCA",
                                           "NumAppointments",
                                           "TimePeriod",
                                           "OwnerType"))
shap.plot.summary(data_long = shap_values2,
                  kind = "bar")



# Feature-Value plot -------------

# train_data or balanced_data
train_final <- balanced_data %>%
  select(-apptdate, -key) %>%
  encodeXGBoost() %>%
  mutate(PregnantNR = ifelse(GenderFemale == 1 & between(Age, 10, 49), 0, PregnantNR),
         BreastfeedingNR = ifelse(GenderFemale == 1 & between(Age, 10, 49), 0, BreastfeedingNR)) %>%
  as.data.frame()

train_final <- train_final %>%
  rename("DaystoNextAppointment" = NextAppointmentDate,
         "AverageTCA" = average_tca_last5,
         "AverageLateness" = averagelateness,
         "AverageLatenessLast10" = averagelateness_last10,
         "AverageLatenessLast5" = averagelateness_last5,
         "AverageLatenessLast3" = averagelateness_last3,
         "Circumcision" = circumcision,
         "LateRate" = late_rate,
         "NumAppointments" = n_appts,
         "Owner Type - Faith" = owner_typeFaith,
         "Owner Type - NGO" = owner_typeNGO,
         "TimeonART" = timeOnArt,
         "UnscheduledRate" = unscheduled_rate,
         "LateLast1" = visit_1)

dtrain <- xgb.DMatrix(data = data.matrix(train_final[,which(names(train_final) != "Target")]),
                      label = train_final$Target)

set.seed(2231)
xgb <- xgboost::xgb.train(data = dtrain,
                          eta = 0.01,
                          max_depth = 8,
                          colsample_bytree = 0.5,
                          nrounds = 500,
                          objective = "binary:logistic",
                          # metric = 'aucpr',
                          verbose = 0
)


shap_values <- shap.values(xgb_model = xgb,
                           X_train = data.matrix(train_final[,which(names(train_final) != "Target")]))

shap.plot.summary.wrap2(shap_score = shap_values$shap_score,
                        X = data.matrix(train_final[,which(names(train_final) != "Target")]),
                        top_n = 15)

# Calculate feature importance
importance_matrix <- xgb.importance(colnames(dtrain), model = xgb)

# Create a ggplot graph of feature importance
library(ggplot2)
ggplot(importance_matrix, aes(x = reorder(Feature, -Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(x = "Feature", y = "Importance", title = "Feature Importance") +
  theme_minimal()