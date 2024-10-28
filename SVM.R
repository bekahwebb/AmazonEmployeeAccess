library(tidyverse)
library(DataExplorer)
library(vroom)
library(ggplot2)
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
library('kernlab')

amazon_test <- vroom("./test.csv")
#amazon_test
amazon_train <- vroom("./train.csv")
#amazon_train

# turn ACTION into a factor
amazon_train$ACTION <- as.factor(amazon_train$ACTION)

sweet_recipe <- recipe(ACTION~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .00001) %>% # combines rare categories that occur less often
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))%>% #target encoding
  step_normalize(all_predictors())%>%
  step_pca(all_predictors(), threshold=0.9)#Threshold is between 0 and 1
  #step_nzv(all_predictors())  # Add this step to filter out near-zero variance features

# apply the recipe to your data
prep_recipe <- prep(sweet_recipe)
baked <- bake(prep_recipe, new_data = amazon_train)
# Check if there are any NAs in the processed dataset
#sum(is.na(baked))
sweet_recipe

#linear kernel model
svmLinear <- svm_linear(cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

#set workflow
svmLinear_wf <- workflow() %>%
  add_recipe(sweet_recipe) %>%
  add_model(svmLinear)

## Fit or Tune Model HERE
## Grid of values to tune over
   
tuning_grid_linear <- grid_regular(cost(), 
                                   levels = 5)

## Split data for CV
folds <- vfold_cv(amazon_train, v = 5, repeats=1)

## Run the CV

CV_results_linear <- svmLinear_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid_linear,
            metrics=metric_set(roc_auc, f_meas, sens, recall, spec,
                               precision, accuracy)) #Or leave metrics NULL

## Find Best Tuning Parameters

bestTune <- CV_results_linear %>%
  select_best(metric="roc_auc")

## Finalize the Workflow & fit it

final_wf <-
  svmLinear_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazon_train)

## Predict Linear
amazon_predictions <- final_wf %>%
  predict(new_data =amazon_test,
          type="prob")


## Format the Predictions for Submission to Kaggle
svmLinear_kaggle_submission <- amazon_predictions%>%
  rename(ACTION=.pred_1) %>%
  select(ACTION) %>%
  bind_cols(., amazon_test) %>% #Bind predictions with test data
  select(id, ACTION)  #keep Id, ACTION for submission


## Write out the file
vroom_write(x=svmLinear_kaggle_submission, file="svmLinearPreds.csv", delim=",")
#public .86359 private .85632, it took about 5 and a half min. to run on batch.
#went from threshold = .00001 to threshold = .001 to see if that changes anything.
#it got stuck at the cross validation in the batch and then killed it, and stopped running it.  
#I'll stick with 0.86359
