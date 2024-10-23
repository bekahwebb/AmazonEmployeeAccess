library(tidyverse)
library(DataExplorer)
library(vroom)
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
library(discrim)
library('ggplot2')
library('naivebayes')


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

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
set_mode("classification") %>%
set_engine("naivebayes") # install discrim library for the naivebayes

## Create a workflow with model & recipe

nb_wf <- workflow() %>%
  add_recipe(sweet_recipe) %>%
  add_model(nb_model)

## Set up grid of tuning values
tuning_grid <- grid_regular(Laplace(), smoothness(),levels = 3)

## Set up K-fold CV
folds <- vfold_cv(amazon_train, v = 5, repeats=1)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc, f_meas, sens, recall, spec,
                               precision, accuracy)) 

## Find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric="roc_auc")

## Finalize workflow and predict
final_wf <-
  nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazon_train)

## Predict
nb_amazon_predictions <- final_wf %>%
  predict(new_data =amazon_test, type="prob")


## Format the Predictions for Submission to Kaggle
naivebayes_kaggle_submission <- nb_amazon_predictions%>%
  rename(ACTION=.pred_1) %>%
  select(ACTION) %>%
  bind_cols(., amazon_test) %>% #Bind predictions with test data
  select(id, ACTION)  #keep Id, ACTION for submission


## Write out the file
vroom_write(x=naivebayes_kaggle_submission, file="nbPreds.csv", delim=",")
#public score 0.85424, private score 0.84535 cutoff is a .885
#adding 3 levels I got .85682 for public, and 0.84804 for private