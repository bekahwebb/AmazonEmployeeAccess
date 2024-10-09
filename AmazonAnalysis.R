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

amazon_test <- vroom("C:/Users/bekah/Downloads/amazon-employee-access-challenge/test.csv")
amazon_test
amazon_train <- vroom("C:/Users/bekah/Downloads/amazon-employee-access-challenge/train.csv")
amazon_train
amazon_sample <- vroom("C:/Users/bekah/Downloads/amazon-employee-access-challenge/sampleSubmission.csv")
amazon_sample

# turn ACTION into a factor
amazon_train$ACTION <- as.factor(amazon_train$ACTION)

#EDA
variable_plot <- DataExplorer::plot_intro(amazon_train) # visualization of glimpse! 
variable_plot

correlation_plot <- DataExplorer::plot_correlation(amazon_train) # correlation heat map 
correlation_plot
# 
# #plots 2x2 tables

plot2 <- ggplot(data=amazon_train) + geom_mosaic(aes(x=product(MGR_ID), fill=ACTION))
plot2
plot3 <- ggplot(amazon_train, aes(x = ROLE_TITLE, y = ACTION)) +
  geom_boxplot() +
  labs(title = "Access Levels by Role Title", x = "Role Title", y = "Action")
plot4 <-ggplot(amazon_train, aes(x = ACTION)) +
  geom_bar(fill = "turquoise") +
  labs(title = "Distribution of ACTION", x = "ACTION", y = "Frequency")

plot3
plot4


# plot1 + plot2 #side by side
plot3 / plot4 #stacked


# Feature Engineering
sweet_recipe <- recipe(ACTION~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .01) %>% # combines rare categories that occur less often
  step_dummy(all_nominal_predictors())%>%
  step_normalize(all_predictors())


# NOTE: some of these step functions are not appropriate to use together

# apply the recipe to your data
prep_recipe <- prep(sweet_recipe)
baked <- bake(prep_recipe, new_data = amazon_train)
baked
ncol(baked)
#1050 column names
colnames(baked)
colnames(amazon_train)
#10 column names [1] "ACTION"           "RESOURCE"         "MGR_ID"           "ROLE_ROLLUP_1"    "ROLE_ROLLUP_2"   
#[6] "ROLE_DEPTNAME"    "ROLE_TITLE"       "ROLE_FAMILY_DESC" "ROLE_FAMILY"      "ROLE_CODE"  

#logistic regression 10/9/24

logRegModel <- logistic_reg() %>% #Type of model
  set_engine("glm")

## Put into a workflow here
## Combine into a Workflow and fit
logReg_workflow <- workflow() %>% #sets up a series of steps that you can apply to any dataset
  add_recipe(sweet_recipe) %>%
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
  #.70429 public .69688
