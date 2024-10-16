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
#10/16/24 Knn
## knn model
knn_model <- nearest_neighbor(neighbors=tune()) %>% # tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(sweet_recipe) %>%
  add_model(knn_model)


## Fit or Tune Model HERE

## Grid of values to tune over
tuning_grid <- grid_regular(neighbors(range = c(1, 30)),
                            levels = 5) 

## Split data for CV
folds <- vfold_cv(amazon_train, v = 5, repeats=1)

## Run the CV
CV_results <- knn_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc, f_meas, sens, recall, spec,
                               precision, accuracy)) #Or leave metrics NULL
## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric="roc_auc")

## Finalize the Workflow & fit it
knn_wf <-
  knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazon_train)

## Predict
knn_amazon_predictions <- knn_wf %>%
  predict(new_data =amazon_test,
          type="prob")

## Format the Predictions for Submission to Kaggle
knn_logistic_kaggle_submission <- knn_amazon_predictions%>%
  rename(ACTION=.pred_1) %>%
  select(ACTION) %>%
  bind_cols(., amazon_test) %>% #Bind predictions with test data
  select(id, ACTION)  #keep Id, ACTION for submission


## Write out the file
vroom_write(x=knn_logistic_kaggle_submission, file="knnPreds.csv", delim=",")
#kaggle score 0.79621