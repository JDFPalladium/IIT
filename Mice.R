library(dplyr)
library(caret)
library(xgboost)
library(PRROC)
library(data.table)
library(aws.s3)
library(mice)

Sys.setenv("AWS_ACCESS_KEY_ID" = "",
           "AWS_SECRET_ACCESS_KEY" = "",
           "AWS_DEFAULT_REGION" = "")

# name the bucket
aws_bucket <- ""

# get the bucket
get_bucket(aws_bucket)

uploads <- s3read_using(FUN = read.csv, bucket = aws_bucket, object = "Latest_Uploads_2024_Feb2024 1.csv")
uploads$DateReceived <- ymd(gsub( " .*$", "", uploads$DateRecieved))
uploads <- uploads %>% select(SiteCode, DateReceived)
uploads$SiteCode <- as.character(uploads$SiteCode)



setwd("~/February2024")
file.list <- list.files(pattern='*.rds')
file.list <- file.list[grep("prep_0314", file.list)]
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
  select(-PatientSource, -visit_2, -visit_3, -visit_4, -visit_5, -BMI, -Weight)#,
         # -most_recent_vl, -n_tests_threeyears, -n_hvl_threeyears, -n_lvl_threeyears,
         # -n_appts, -late, -late28, DifferentiatedCare, timeOnArt,
         # -late_last10, averagelateness_last10,
         # -recent_hvl_rate, -AHD
  # )

gis <- read.csv('../gis_features_dec2023.csv') %>%
  dplyr::select(-Latitude, -Longitude) #%>%
# mutate(FacilityCode = as.character(FacilityCode))
iit <- merge(iit, gis,by.x = "SiteCode", by.y = "FacilityCode", all.x = TRUE) %>%
  dplyr::select(-SiteCode)

iit$apptdate <- iit$PredictionDate + iit$NextAppointmentDate

iit$yr <- year(iit$apptdate)
iit$mt <- month(iit$apptdate)

iit$SiteCode <- substr(iit$key, nchar(iit$key)-4, nchar(iit$key))
iit <- merge(iit, uploads, by = "SiteCode")

iit$cutoff_date <- iit$apptdate + 30
iit$keep <- ifelse(iit$cutoff_date < iit$DateReceived, 1, 0)
iit <- iit[iit$keep == 1,]
iit <- iit %>%
  filter(!(yr == "2023" & mt == "12"))

iit <- iit %>%
  select(-keep, - DateReceived, -PredictionDate, -cutoff_date, -SiteCode, -yr, -mt)
saveRDS(iit, "iit_0715_prepped.rds")

setwd("~/February2024")
iit <- readRDS("iit_0715_prepped.rds")

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
  filter(apptdate <= "2022-12-31") 

test_data <- iit %>% 
  ungroup() %>%
  filter(patient_group == 2) %>%
  select(-patient_group) %>%
  arrange(apptdate) %>%
  filter(apptdate >= "2023-01-01") %>%
  filter(apptdate <= "2023-11-30") %>%
  select(-apptdate, - key) 

# # Create folds
# nfolds <- 10
# cuts <- seq(min(train_data$apptdate), max(train_data$apptdate), length.out = nfolds + 2)
# 
# 
# # Train MICE
# for (j in 1:nfolds) {
#   print(j)
#   set.seed(2231)
#   tmp <-  train_data %>%
#     group_by(key) %>% 
#     mutate(patient_group = sample(1:2, 1, prob = c(0.7, 0.3))) 
#   
#   train_tmp <- tmp %>%
#     ungroup() %>%
#     filter(patient_group == 1) %>%
#     filter(between(apptdate, cuts[1], cuts[j+1])) %>%
#     select(-key, -apptdate, -condom, -pregnancies,
#            -pnc, -sba, -anc, -poverty, -circumcision, -Pregnant,
#            -Breastfeeding, -patient_group)
#   train_to_impute <- train_tmp %>% select(-Target)
#   
#   val_tmp <- tmp %>%
#     ungroup() %>%
#     filter(patient_group == 2) %>%
#     filter(between(apptdate, cuts[j+1], cuts[j+2])) %>%
#     select(-key, -apptdate, -condom, -pregnancies,
#            -pnc, -sba, -anc, -poverty, -circumcision, -Pregnant,
#            -Breastfeeding, -patient_group)
#   val_to_impute <- val_tmp %>% select(-Target)
#   
#   train_to_impute <- as.data.frame(train_to_impute)
#   val_to_impute <- as.data.frame(val_to_impute)
#   
#   # Identify character columns
#   char_cols <- sapply(train_to_impute, is.character)
#   # Convert character columns to factors
#   train_to_impute[char_cols] <- lapply(train_to_impute[char_cols], as.factor)
#   val_to_impute[char_cols] <- lapply(val_to_impute[char_cols], as.factor)
#   
#   iit_impute_mice <- mice(train_to_impute,
#                           pred = quickpred(train_to_impute),
#                           m = 5,
#                           maxit = 5,
#                           seed = 2231)
#   
#   iit_impute_mice_train <- mice::complete(iit_impute_mice, action = "broad") # get all imputations
#   
#   # Get variables with missing fields
#   vars_to_impute_mode <- names(train_to_impute)[sapply(train_to_impute,
#                                                        function(x) any(is.na(x)) & is.factor(x))]
#   vars_to_impute_mean <- names(train_to_impute)[sapply(train_to_impute,
#                                                        function(x) any(is.na(x)) & is.numeric(x))]
#   
#   
#   for(l in 1:nrow(train_to_impute)){
#     for(k in vars_to_impute_mode){
#       if(is.na(train_to_impute[l,k])){
#         x <- c(as.character(iit_impute_mice_train[l, paste0(k, ".1")]),
#                as.character(iit_impute_mice_train[l, paste0(k, ".2")]),
#                as.character(iit_impute_mice_train[l, paste0(k, ".3")]),
#                as.character(iit_impute_mice_train[l, paste0(k, ".4")]),
#                as.character(iit_impute_mice_train[l, paste0(k, ".5")]))
#         ux <- unique(x)
#         train_to_impute[l,k] <- ux[which.max(tabulate(match(x, ux)))]
#       }
#     }
#   }
#   
#   # loop through vars_to_impute and get mean for imputation
#   for(k in vars_to_impute_mean){
#     for(l in 1:nrow(train_to_impute)){
#       if(is.na(train_to_impute[l,k])){
#         x <- c(as.numeric(iit_impute_mice_train[l, paste0(k, ".1")]),
#                as.numeric(iit_impute_mice_train[l, paste0(k, ".2")]),
#                as.numeric(iit_impute_mice_train[l, paste0(k, ".3")]),
#                as.numeric(iit_impute_mice_train[l, paste0(k, ".4")]),
#                as.numeric(iit_impute_mice_train[l, paste0(k, ".5")]))
#         train_to_impute[l,k] <- mean(x)
#       }
#     }
#   }
#   
#   # Add label back in
#   train_to_impute$Target <- train_tmp$Target
#   train_to_impute$Cat <- "train"
#   
#   # Extend mice to validation set
#   vals_mice_wide <- mice.mids(iit_impute_mice, newdata = val_to_impute)
#   vals_mice_wide <- complete(vals_mice_wide, action = "broad") # get all imputations
#   
#   # loop through vars_to_impute and get mode
#   for(k in vars_to_impute_mode){
#     for(l in 1:nrow(val_to_impute)){
#       if(is.na(val_to_impute[l,k])){
#         x <- c(as.character(vals_mice_wide[l, paste0(k, ".1")]),
#                as.character(vals_mice_wide[l, paste0(k, ".2")]),
#                as.character(vals_mice_wide[l, paste0(k, ".3")]),
#                as.character(vals_mice_wide[l, paste0(k, ".4")]),
#                as.character(vals_mice_wide[l, paste0(k, ".5")]))
#         ux <- unique(x)
#         val_to_impute[l,k] <- ux[which.max(tabulate(match(x, ux)))]
#       }
#     }
#   }
#   
#   # loop through vars_to_impute and get mean for imputation
#   for(k in vars_to_impute_mean){
#     for(l in 1:nrow(val_to_impute)){
#       if(is.na(val_to_impute[l,k])){
#         x <- c(as.numeric(vals_mice_wide[l, paste0(k, ".1")]),
#                as.numeric(vals_mice_wide[l, paste0(k, ".2")]),
#                as.numeric(vals_mice_wide[l, paste0(k, ".3")]),
#                as.numeric(vals_mice_wide[l, paste0(k, ".4")]),
#                as.numeric(vals_mice_wide[l, paste0(k, ".5")]))
#         val_to_impute[l,k] <- mean(x)
#       }
#     }
#   }
#   
#   # Add label back in
#   val_to_impute$Target <- val_tmp$Target
#   val_to_impute$Cat <- "val"
#   
#   iter_out <- bind_rows(train_to_impute,
#                         val_to_impute)
#   
#   saveRDS(iter_out, paste0("iit_mice_03142024_", j, ".rds"))
#   
# }

train_tmp <- train_data %>%
  ungroup() %>%
  select(-key, -apptdate, -condom, -pregnancies,
         -pnc, -sba, -anc, -poverty, -circumcision, -Pregnant,
         -Breastfeeding)
train_to_impute <- train_tmp %>% select(-Target)

val_tmp <- test_data %>%
  ungroup() %>%
  select(-condom, -pregnancies,
         -pnc, -sba, -anc, -poverty, -circumcision, -Pregnant,
         -Breastfeeding)
val_to_impute <- val_tmp %>% select(-Target)

train_to_impute <- as.data.frame(train_to_impute)
val_to_impute <- as.data.frame(val_to_impute)

# Identify character columns
char_cols <- sapply(train_to_impute, is.character)
# Convert character columns to factors
train_to_impute[char_cols] <- lapply(train_to_impute[char_cols], as.factor)
val_to_impute[char_cols] <- lapply(val_to_impute[char_cols], as.factor)

iit_impute_mice <- mice(train_to_impute,
                        pred = quickpred(train_to_impute),
                        m = 5,
                        maxit = 5,
                        seed = 2231)

iit_impute_mice_train <- mice::complete(iit_impute_mice, action = "broad") # get all imputations

# Get variables with missing fields
vars_to_impute_mode <- names(train_to_impute)[sapply(train_to_impute,
                                                     function(x) any(is.na(x)) & is.factor(x))]
vars_to_impute_mean <- names(train_to_impute)[sapply(train_to_impute,
                                                     function(x) any(is.na(x)) & is.numeric(x))]


for(l in 1:nrow(train_to_impute)){
  for(k in vars_to_impute_mode){
    if(is.na(train_to_impute[l,k])){
      x <- c(as.character(iit_impute_mice_train[l, paste0(k, ".1")]),
             as.character(iit_impute_mice_train[l, paste0(k, ".2")]),
             as.character(iit_impute_mice_train[l, paste0(k, ".3")]),
             as.character(iit_impute_mice_train[l, paste0(k, ".4")]),
             as.character(iit_impute_mice_train[l, paste0(k, ".5")]))
      ux <- unique(x)
      train_to_impute[l,k] <- ux[which.max(tabulate(match(x, ux)))]
    }
  }
}

# loop through vars_to_impute and get mean for imputation
for(k in vars_to_impute_mean){
  for(l in 1:nrow(train_to_impute)){
    if(is.na(train_to_impute[l,k])){
      x <- c(as.numeric(iit_impute_mice_train[l, paste0(k, ".1")]),
             as.numeric(iit_impute_mice_train[l, paste0(k, ".2")]),
             as.numeric(iit_impute_mice_train[l, paste0(k, ".3")]),
             as.numeric(iit_impute_mice_train[l, paste0(k, ".4")]),
             as.numeric(iit_impute_mice_train[l, paste0(k, ".5")]))
      train_to_impute[l,k] <- mean(x)
    }
  }
}

# Add label back in
train_to_impute$Target <- train_tmp$Target
train_to_impute$Cat <- "train"

# Extend mice to validation set
vals_mice_wide <- mice.mids(iit_impute_mice, newdata = val_to_impute)
vals_mice_wide <- complete(vals_mice_wide, action = "broad") # get all imputations

# loop through vars_to_impute and get mode
for(k in vars_to_impute_mode){
  for(l in 1:nrow(val_to_impute)){
    if(is.na(val_to_impute[l,k])){
      x <- c(as.character(vals_mice_wide[l, paste0(k, ".1")]),
             as.character(vals_mice_wide[l, paste0(k, ".2")]),
             as.character(vals_mice_wide[l, paste0(k, ".3")]),
             as.character(vals_mice_wide[l, paste0(k, ".4")]),
             as.character(vals_mice_wide[l, paste0(k, ".5")]))
      ux <- unique(x)
      val_to_impute[l,k] <- ux[which.max(tabulate(match(x, ux)))]
    }
  }
}

# loop through vars_to_impute and get mean for imputation
for(k in vars_to_impute_mean){
  for(l in 1:nrow(val_to_impute)){
    if(is.na(val_to_impute[l,k])){
      x <- c(as.numeric(vals_mice_wide[l, paste0(k, ".1")]),
             as.numeric(vals_mice_wide[l, paste0(k, ".2")]),
             as.numeric(vals_mice_wide[l, paste0(k, ".3")]),
             as.numeric(vals_mice_wide[l, paste0(k, ".4")]),
             as.numeric(vals_mice_wide[l, paste0(k, ".5")]))
      val_to_impute[l,k] <- mean(x)
    }
  }
}

# Add label back in
val_to_impute$Target <- val_tmp$Target
val_to_impute$Cat <- "val"

iter_out <- bind_rows(train_to_impute,
                      val_to_impute)

saveRDS(iter_out, paste0("iit_mice_03142024_full.rds"))