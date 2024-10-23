library(tidyverse)
library(DataExplorer)
library(vroom)
library(ggplot2)
library(tidyverse)
library(patchwork)
library(readr)
library(GGally)
library(poissonreg)
library(recipes)
library(rsample)
library(magrittr)
library(tidymodels)
library(lubridate)
library(poissonreg) #if you want to do penalized, poisson regression
library(rpart)
library(ranger)
library(stacks) # you need this library to create a stacked model
library(embed) # for target encoding
library(ggmosaic)

amazon_test <- vroom("./test.csv")

amazon_train <- vroom("./train.csv")


# turn ACTION into a factor
amazon_train$ACTION <- as.factor(amazon_train$ACTION)

#my recipe
# Feature Engineering
sweet_recipe <- recipe(ACTION~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .00001) %>% # combines rare categories that occur less often
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))%>% #target encoding
  step_normalize(all_predictors())

# turn ACTION into a factor
amazon_train$ACTION <- as.factor(amazon_train$ACTION)
rf_model <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
set_engine("ranger") %>%
set_mode("classification")

## Create a workflow with model & recipe

rf_wf <- workflow() %>%
  add_recipe(sweet_recipe) %>%
  add_model(rf_model)

## Set up grid of tuning values
tuning_grid <- grid_regular(mtry(range = c(1, 10)),
                            min_n(),
                            levels = 3) 

## Set up K-fold CV
folds <- vfold_cv(amazon_train, v = 5, repeats=1)

## Run the CV
CV_results <- rf_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc, f_meas, sens, recall, spec,
                               precision, accuracy)) 

## Find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric="roc_auc")

## Finalize workflow and predict
final_wf <-
  rf_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazon_train)

## Predict
rf_amazon_predictions <- final_wf %>%
  predict(new_data =amazon_test, type="prob")


## Format the Predictions for Submission to Kaggle
rf_kaggle_submission <- rf_amazon_predictions%>%
  rename(ACTION=.pred_1) %>%
  select(ACTION) %>%
  bind_cols(., amazon_test) %>% #Bind predictions with test data
  select(id, ACTION)  #keep Id, ACTION for submission


## Write out the file
vroom_write(x=rf_kaggle_submission, file="rfPreds.csv", delim=",")
#public score .88523 private score .87370, it took 486 seconds or about 8 minutes to run on batch