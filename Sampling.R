library(aws.s3)
library(dplyr)

Sys.setenv("AWS_ACCESS_KEY_ID" = "",
           "AWS_SECRET_ACCESS_KEY" = "",
           "AWS_DEFAULT_REGION" = "")

# name the bucket
aws_bucket <- ""

# get the bucket
get_bucket(aws_bucket)

# Let's get patients who are either active or LTFU - 1,510,096 of these
fact <- s3read_using(FUN = read.csv, bucket = aws_bucket, object = "FEB01 FactART.csv")
fact$key <- paste0(fact$PatientPK, fact$PatientId, fact$SiteCode)
fact_iit_active <- filter(fact, ARTOutcomeDescription %in% c("ACTIVE", "LOSS TO FOLLOW UP"))
# Only 1,466,233 unique combos - let's take ones that are unique
n_distinct(fact_iit_active$key)
fact_iit_active <- fact_iit_active[!duplicated(fact_iit_active$key),]
fact_iit_active <- filter(fact_iit_active, !is.na(SiteCode))


# For VL, filter for keys that are in both lab and visits, and then sample 200K patients

# Now let's get a sample of 200,000
mfl_probs <- fact_iit_active %>%
  group_by(SiteCode) %>%
  summarize(count = n()) %>%
  ungroup() %>%
  mutate(prob = count / sum(count))

fact_iit_active <- merge(fact_iit_active, 
                         mfl_probs, 
                         by = "SiteCode")

set.seed(2231)
keys <- sample(fact_iit_active$key, 300000, replace = FALSE, prob = fact_iit_active$prob)

# create function for writing to S3
writeToS3 = function(file, bucket, filename){
  s3write_using(file, FUN = write.csv,
                bucket = bucket,
                object = filename)
}

writeToS3(keys, aws_bucket, 'keys.csv')

fact_iit_active <- fact_iit_active[fact_iit_active$key %in% keys, ]
writeToS3(fact_iit_active, aws_bucket, 'dem_sample.csv')

# Subset visits
visits <- s3read_using(FUN = read.csv, bucket = aws_bucket, object = "CT_PatientVisits JAN31.csv")
visits$key <- paste0(visits$PatientPK, visits$PatientID, visits$Sitecode)
visits_samp <- visits[visits$key %in% keys, ]
writeToS3(visits_samp, aws_bucket, 'visits_sample.csv')

# Subset pharmacy
pharmacy <- s3read_using(FUN = read.csv, bucket = aws_bucket, object = "CT_PatientPharmacy JAN31.csv")
pharmacy$key <- paste0(pharmacy$PatientPK, pharmacy$PatientID, pharmacy$SiteCode)
pharmacy_samp <- pharmacy[pharmacy$key %in% keys, ]
writeToS3(pharmacy_samp, aws_bucket, 'pharmacy_sample.csv')

# Subset lab
lab <- s3read_using(FUN = read.csv, bucket = aws_bucket, object = "CT_PatientLabs JAN31.csv")
lab$key <- paste0(lab$PatientPk, lab$PatientID, lab$SiteCode)
lab_samp <- lab[lab$key %in% keys, ]
writeToS3(lab_samp, aws_bucket, 'lab_sample.csv')

