library(dplyr)
library(caret)
library(xgboost)
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

uploads <- s3read_using(FUN = read.csv, bucket = aws_bucket, object = "Latest_Uploads_2024_Feb2024 1.csv")
uploads$DateReceived <- ymd(gsub( " .*$", "", uploads$DateRecieved))
uploads <- uploads %>% select(SiteCode, DateReceived)
uploads$SiteCode <- as.character(uploads$SiteCode)

# Bottom line variables that drift
# 1) differentiated care
# 2) num_regimens
# 3) n_appts - drop?
# 4) late - drop?
# 5) late28 - drop?
# 6) timeOnART


setwd("~/February2024")
file.list <- list.files(pattern='*.rds')
file.list <- file.list[grep("0313", file.list)]
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
         # -n_appts, -late, -late28, DifferentiatedCare, timeOnArt,
         # -late_last10, averagelateness_last10,
         -recent_hvl_rate, -AHD
  )

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
# iit <- iit %>% filter(keep == 1)

a <- iit %>%
  filter(!(yr == "2023" & mt == "12")) %>%
  group_by(yr, mt) %>% 
  summarize(iit = mean(Target), count = n())
b <- iit %>%
  filter(keep == 1) %>%
  filter(!(yr == "2023" & mt == "12")) %>%
  group_by(yr, mt) %>%
  summarize(iit = mean(Target), count = n())
d <- merge(a, b, by =c("yr", "mt")) %>% arrange(yr, mt) %>%
  mutate(tp = row_number()) %>%
  filter(!(yr == "2023" & mt == "12"))
plot(d$tp, d$iit.x)
plot(d$tp, d$iit.y)

b$TimePeriod <- as.Date(paste(b$yr, b$mt, "01", sep = "-"), format = "%Y-%m-%d")

# Create graph
ggplot(b, aes(x = TimePeriod, y = iit*100)) +
  geom_line(color = "blue", size = 1) +     # Line plot
  geom_point(color = "red", size = 2) +     # Points on the line
  labs(title = "IIT in 2022-2023",     # Add title
       x = "Time Period",                         # Label for x-axis
       y = "IIT Rate") +                         # Label for y-axis
  theme_minimal() +                         # Minimal theme
  theme(plot.title = element_text(hjust = 0.5)) + # Center the title
  scale_x_date(date_labels = "%b %Y",       
               date_breaks = "3 month") 

# Linear regression to see if tp is statisticall significant
b <- b %>%
  ungroup() %>%
  mutate(tp = row_number())

mod <- lm(iit ~ tp, data = b)

# Target
View(iit %>% group_by(yr, mt) %>% summarize(iit = mean(Target)))

# Categorical features
# decently stable
View(iit %>% group_by(yr, mt) %>% summarize(fem_share = sum(Gender=="Female")/n()))

# decently stable
View(iit %>% group_by(yr, mt) %>% summarize(fem_divorced = sum(MaritalStatus=="Divorced")/n()))
View(iit %>% group_by(yr, mt) %>% summarize(fem_widow = sum(MaritalStatus=="Widow")/n()))

# decently stable
View(iit %>% group_by(yr, mt) %>% summarize(fem_opt = sum(OptimizedHIVRegimen=="Yes", na.rm = T)/n()))

# decently stable
View(iit %>% group_by(yr, mt) %>% summarize(avg_preg = sum(Pregnant=="yes", na.rm = T)/n()))

# decently stable
View(iit %>% group_by(yr, mt) %>% summarize(avg_preg = sum(Breastfeeding=="yes", na.rm = T)/n()))

# increasing trend
View(iit %>% group_by(yr, mt) %>% summarize(avg = sum(DifferentiatedCare=="fasttrack", na.rm = T)/n()))
# increasing trend
View(iit %>% group_by(yr, mt) %>% summarize(avg = sum(DifferentiatedCare=="express", na.rm = T)/n()))
# decreasing trend
View(iit %>% group_by(yr, mt) %>% summarize(avg = sum(DifferentiatedCare=="standardcare", na.rm = T)/n()))

# stable
View(iit %>% group_by(yr, mt) %>% summarize(avg = sum(StabilityAssessment=="Stable", na.rm = T)/n()))



# Numeric Features
# Clear upward trend
View(iit %>% filter(keep == 1) %>% group_by(yr, mt) %>% summarize(avg_appts = mean(n_appts)))
# Does IIT decrease as n_appts increases? Is that why there's a drop?

# Increasing over time
View(iit %>% group_by(yr, mt) %>% summarize(avg_late = mean(late)))

# Increasing over time
View(iit %>% group_by(yr, mt) %>% summarize(avg_late28 = mean(late28)))

# Decently stable over time
View(iit %>% group_by(yr, mt) %>% summarize(avg_lateness = mean(averagelateness, na.rm = T)))

# Decently stable over time
View(iit %>% group_by(yr, mt) %>% summarize(avg_laterate = mean(late_rate, na.rm = T)))

# Decently stable over time
View(iit %>% group_by(yr, mt) %>% summarize(avg_late28_rate = mean(late28_rate, na.rm = T)))

# Decently stable over time
View(iit %>% group_by(yr, mt) %>% summarize(avg_latelast5 = mean(late_last5, na.rm = T)))

# Decently stable over time
View(iit %>% group_by(yr, mt) %>% summarize(avg_latenesslast5 = mean(averagelateness_last5, na.rm = T)))

# Decently stable over time
View(iit %>% group_by(yr, mt) %>% summarize(avg_latelast3 = mean(late_last3, na.rm = T)))

# Decently stable over time
View(iit %>% group_by(yr, mt) %>% summarize(avg_latenesslast3 = mean(averagelateness_last3, na.rm = T)))

# Increasing slightly over time
View(iit %>% group_by(yr, mt) %>% summarize(avg_latelast10 = mean(late_last10, na.rm = T)))

# Decently stable over time
View(iit %>% group_by(yr, mt) %>% summarize(avg_latenesslast10 = mean(averagelateness_last10, na.rm = T)))

# Decently stable over time
View(iit %>% group_by(yr, mt) %>% summarize(avg_visit1 = mean(visit_1, na.rm = T)))

# Decently stable over time
View(iit %>% group_by(yr, mt) %>% summarize(avg_nad = mean(NextAppointmentDate, na.rm = T)))

# maybe a trend?
View(iit %>% group_by(yr, mt) %>% summarize(avg_tca = mean(average_tca_last5, na.rm = T)))

# slight downward trend
View(iit %>% group_by(yr, mt) %>% summarize(avg_reg = mean(num_hiv_regimens, na.rm = T)))

# Decently stable over time
View(iit %>% group_by(yr, mt) %>% summarize(avg_sched = mean(unscheduled_rate, na.rm = T)))

# increasing trend
View(iit %>% group_by(yr, mt) %>% summarize(avg_time = mean(timeOnArt, na.rm = T)))

# stable over time
View(iit %>% group_by(yr, mt) %>% summarize(avg_age = mean(Age, na.rm = T)))



