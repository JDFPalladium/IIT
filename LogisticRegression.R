library(dplyr)
library(caret)
library(ranger)
library(PRROC)
library(data.table)

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
mod <- glm("Target ~ .",
           family = "binomial",
           data = train_final)

val_predict <- predict(mod, newdata = test_data, type = "response")
fg <- val_predict[test_data$Target == 1]
bg <- val_predict[test_data$Target == 0]
prc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
