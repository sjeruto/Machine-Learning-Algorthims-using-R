rm(list=ls())

library(caret)


# Read in and format our data
heart_failure <- read.csv("heart_failure.csv")
str(heart_failure)

heart_failure$DEATH_EVENT <- as.factor(heart_failure$DEATH_EVENT)

levels(heart_failure$DEATH_EVENT)

levels(heart_failure$DEATH_EVENT) <- c("No","Yes")

target_col <- grep("DEATH_EVENT",names(heart_failure))

set.seed(1729) #TRIVIA: What's special about 1729?
# Generate test and training sets
train_indices = createDataPartition(y = heart_failure$DEATH_EVENT, p = 0.8, list = F)
training = heart_failure[train_indices , ]
testing = heart_failure[-train_indices, ]
nrow(training)
nrow(testing)

trCtrl <- trainControl(method = "cv",
                       number = 5,
                       search= "grid",
                       summaryFunction = twoClassSummary,
                       classProbs = TRUE)

trGrid <- expand.grid(size =1:8, decay = c(0.1,0.05,0.025,0))

nn_model <- caret::train(DEATH_EVENT ~., 
                         data = training, 
                         trControl = trCtrl,
                         method="nnet",
                         metric="ROC",
                         tuneGrid = trGrid,
                         preProcess = c('center','scale')) 

plot(nn_model)
print(nn_model)

train_predictions <- predict(nn_model,training[,-target_col])

cbind(train_predictions, training$DEATH_EVENT)

#confusion matrix

table(predict=train_predictions,actual=training$DEATH_EVENT)

test_predictions <- predict(nn_model,testing[,-target_col])

cbind(test_predictions, testing$DEATH_EVENT)

#confusion matrix

table(predict=test_predictions,actual=testing$DEATH_EVENT)
