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
library('themis')
library(discrim)

amazon_test <- vroom("./test.csv")
#amazon_test
amazon_train <- vroom("./train.csv")
#amazon_train

# turn ACTION into a factor
amazon_train$ACTION <- as.factor(amazon_train$ACTION)

zen_recipe <- recipe(ACTION~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .00001) %>% # combines rare categories that occur less often
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))%>% #target encoding
  step_normalize(all_predictors())%>%
  # step_pca(all_predictors(), threshold=0.9)%>% #Threshold is between 0 and 1
  step_smote(all_outcomes(), neighbors=3)
  # step_upsample()%>% use all_outcomes
  # step_downsample()

# apply the recipe to your data
prep_recipe <- prep(zen_recipe)
baked <- bake(prep_recipe, new_data = amazon_train)
# Check if there are any NAs in the processed dataset
#sum(is.na(baked))
zen_recipe

logRegModel <- logistic_reg() %>% #Type of model
  set_engine("glm")

## Combine into a Workflow and fit
logReg_workflow <- workflow() %>% #sets up a series of steps that you can apply to any dataset
  add_recipe(zen_recipe) %>%
  add_model(logRegModel) %>%
  fit(data=amazon_train)#fit the workflow

## Make predictions
amazon_predictions <- predict(logReg_workflow,
                              new_data=amazon_test,
                              type="prob") # "class" or "prob" (see doc)

## Format the Predictions for Submission to Kaggle
logistic_kaggle_submission <- amazon_predictions%>%
  rename(ACTION=.pred_1) %>%
  select(ACTION) %>%
  bind_cols(., amazon_test) %>% #Bind predictions with test data
  select(id, ACTION)  #keep Id, ACTION for submission

## Write out the file
vroom_write(x=logistic_kaggle_submission, file="logisticPreds.csv", delim=",")
#.70429 public .69688 before pca step
# improved scores big time to .86377 public .85522 private after pca step, woah!
#.86122 public .85199 private with smote, went down a little
#with just smote and no pca with public score is .86156 and private is .85234 so it improved a little
#from the pca model and a lot from the first model.

#penalized logistic regression 10/11/24
penalize_logistic_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
  set_engine("glmnet")

amazon_workflow <- workflow() %>%
  add_recipe(zen_recipe) %>%
  add_model(penalize_logistic_mod)

## Grid of values to tune over10
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV15
folds <- vfold_cv(amazon_train, v = 5, repeats=1)

## Run the CV
CV_results <- amazon_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc, f_meas, sens, recall, spec,
                               precision, accuracy)) #Or leave metrics NULL
## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric="roc_auc")

# penalty mixture .config
#<dbl>   <dbl> <chr>
# 1) 0.00316     0.5 Preprocessor1_Model14

## Finalize the Workflow & fit it
final_wf <-
  amazon_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazon_train)

## Predict
penalized_amazon_predictions <- final_wf %>%
  predict(new_data =amazon_test,
          type="prob")

## Format the Predictions for Submission to Kaggle
penalized_logistic_kaggle_submission <- penalized_amazon_predictions%>%
  rename(ACTION=.pred_1) %>%
  select(ACTION) %>%
  bind_cols(., amazon_test) %>% #Bind predictions with test data
  select(id, ACTION)  #keep Id, ACTION for submission

## Write out the file
vroom_write(x=penalized_logistic_kaggle_submission, file="penalizedlogisticPreds.csv", delim=",")
#public score .78320 with threshold at .001
#public score .86225 with threshold at .00001
#public with step pca slightly improved to .86842, private .86048 
#public with  a slight decline with smote to 0.86428 and private score of 0.85503
#went up with just smote to public .87389 and private .86616

## knn model
knn_model <- nearest_neighbor(neighbors=tune()) %>% # tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(zen_recipe) %>%
  add_model(knn_model)


## Fit or Tune Model HERE
## Grid of values to tune over
tuning_grid <- grid_regular(neighbors(range(1,30)),
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
final_wf <-
  knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazon_train)

## Predict
knn_amazon_predictions <- final_wf %>%
  predict(new_data =amazon_test,
          type="prob")

## Format the Predictions for Submission to Kaggle
knn_kaggle_submission <- knn_amazon_predictions%>%
  rename(ACTION=.pred_1) %>%
  select(ACTION) %>%
  bind_cols(., amazon_test) %>% #Bind predictions with test data
  select(id, ACTION)  #keep Id, ACTION for submission


## Write out the file
vroom_write(x=knn_kaggle_submission, file="knnPreds.csv", delim=",")
#kaggle score 0.79621 public, private is 0.79084
#went down after pca for public .78319, and private .78602 when the tuning was set to range from 1 to 30
#10/25/24 went to public score of .75518 and a private score of .75817 after leaving the tuning of k open
#after adding step pca it did worse, and went down to a public score .73577 and private score .74841
#10/30/24 with smote public 0.7454, private 0.75766
#added range to the tune for k from 1 to 30 and it improved with the balance step with smote
#public .80399 private .79727

# rf_model <- rand_forest(mtry = tune(),
#                         min_n=tune(),
#                         trees=500) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# ## Create a workflow with model & recipe
# 
# rf_wf <- workflow() %>%
#   add_recipe(zen_recipe) %>%
#   add_model(rf_model)
# 
# ## Set up grid of tuning values
# tuning_grid <- grid_regular(mtry(range = c(1, 10)),
#                             min_n(),
#                             levels = 3) 
# 
# ## Set up K-fold CV
# folds <- vfold_cv(amazon_train, v = 5, repeats=1)
# 
# ## Run the CV
# CV_results <- rf_wf %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(roc_auc, f_meas, sens, recall, spec,
#                                precision, accuracy)) 
# 
# ## Find best tuning parameters
# bestTune <- CV_results %>%
#   select_best(metric="roc_auc")
# 
# ## Finalize workflow and predict
# final_wf <-
#   rf_wf %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=amazon_train)
# 
# ## Predict
# rf_amazon_predictions <- final_wf %>%
#   predict(new_data =amazon_test, type="prob")
# 
# 
# ## Format the Predictions for Submission to Kaggle
# rf_kaggle_submission <- rf_amazon_predictions%>%
#   rename(ACTION=.pred_1) %>%
#   select(ACTION) %>%
#   bind_cols(., amazon_test) %>% #Bind predictions with test data
#   select(id, ACTION)  #keep Id, ACTION for submission
# 
# 
# ## Write out the file
# vroom_write(x=rf_kaggle_submission, file="rfPreds.csv", delim=",")
# #public score .88523 private score .87370, it took 486 seconds or about 8 minutes to run on batch
# #did worse with pca .85114 for public, private is .84141
# #did worse with smote from original model, and slightly better than pca with 0.85429 and 0.84602 private

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes

## Create a workflow with model & recipe

nb_wf <- workflow() %>%
  add_recipe(zen_recipe) %>%
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
# a slight improvement with the pca step to a public score of 0.86659, and a private score of 0.85554
# a slight decline in kaggle scores with smote with a public score of 0.85437 and a private score of 0.84599
#with only smote the public score is .85003, and private is .84270 so it went down with smote

# #linear kernel model
# svmLinear <- svm_linear(cost=tune()) %>% # set or tune
#   set_mode("classification") %>%
#   set_engine("kernlab")
# 
# #set workflow
# svmLinear_wf <- workflow() %>%
#   add_recipe(zen_recipe) %>%
#   add_model(svmLinear)
# 
# ## Fit or Tune Model HERE
# ## Grid of values to tune over
# 
# tuning_grid_linear <- grid_regular(cost(), 
#                                    levels = 5)
# 
# ## Split data for CV
# folds <- vfold_cv(amazon_train, v = 5, repeats=1)
# 
# ## Run the CV
# 
# CV_results_linear <- svmLinear_wf %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid_linear,
#             metrics=metric_set(roc_auc, f_meas, sens, recall, spec,
#                                precision, accuracy)) #Or leave metrics NULL
# 
# ## Find Best Tuning Parameters
# 
# bestTune <- CV_results_linear %>%
#   select_best(metric="roc_auc")
# 
# ## Finalize the Workflow & fit it
# 
# final_wf <-
#   svmLinear_wf %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=amazon_train)
# 
# ## Predict Linear
# amazon_predictions <- final_wf %>%
#   predict(new_data =amazon_test,
#           type="prob")
# 
# 
# ## Format the Predictions for Submission to Kaggle
# svmLinear_kaggle_submission <- amazon_predictions%>%
#   rename(ACTION=.pred_1) %>%
#   select(ACTION) %>%
#   bind_cols(., amazon_test) %>% #Bind predictions with test data
#   select(id, ACTION)  #keep Id, ACTION for submission
# 
# 
# ## Write out the file
# vroom_write(x=svmLinear_kaggle_submission, file="svmLinearPreds.csv", delim=",")
# #public .86359 private .85632, it took about 5 and a half min. to run on batch.
# #went from threshold = .00001 to threshold = .001 to see if that changes anything.
# #it got stuck at the cross validation in the batch and then killed it, and stopped running it.  
# #I'll stick with 0.86359
# #got the same score with the smote step public 0.86359 and 0.85632 score for private
#it took about 13 minutes to run on batch.  I commented out random forest and svm because I ran rf separate
#without the pca step and could see it did better than with pca but still slightly worse with the smote step,
#I need the pca step for the svm so I didn't check the svm model with only smote so I am just checking logreg,
#penalizedreg, knn, and bayes with only the smote step to see how they do, fc
#
