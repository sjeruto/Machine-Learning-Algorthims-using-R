#### Random Forest ####

# Clear everything
rm(list=ls())

# Set a random seed to ensurre reproducibility
set.seed(42) 

# Load libraries if you haven't done so
library (rpart)
library(rpart.plot)
library(mlbench)
library(randomForest)
library(caret)

##Check working directory
getwd() 
setwd("C:/Users/sharon1/Downloads/")

# Read in dataset
wines<-read.csv("C:/Users/sharon1/Downloads/wines.csv")

# Drop the ID column
wines$ID <- NULL

# Let's make this a binary classification
wines$quality_binary <- 1
wines[wines$quality < 7, "quality_binary"] <- 0
wines$quality_binary <- as.factor(wines$quality_binary)

# Need to drop the quality predictor
wines$quality <- NULL

# Explore the dataset
str(wines)
dim(wines)
summary(wines)
# http://archive.ics.uci.edu/ml/datasets/Wine+Quality

# Set up Training and Test sets
trainset_size <- floor(0.80 * nrow(wines))
trainset_indices <- sample(seq_len(nrow(wines)), size = trainset_size)
trainset <- wines[trainset_indices, ]
testset <- wines[-trainset_indices, ]

# Rowcounts to check
nrow(trainset)
nrow(testset)
nrow(wines)

#### Decision tree for comparison ####

# Build model
rpart_model <- rpart(quality_binary ~.,data = trainset, method="class")

# Plot tree
prp(rpart_model)

# Get predictions
rpart_predict <- predict(rpart_model,testset[,-ncol(wines)],type="class")

# Confusion matrix
table(rpart_predict,testset$quality_binary)

# Accuracy for test set
mean(rpart_predict==testset$quality_binary)

#### Random Forest ####

# Build random forest model
wine_rf <- randomForest( quality_binary~.,data = trainset, 
                         importance=TRUE, xtest=testset[,-ncol(wines)],ntree=100) 


# What other things could we set? 
# https://www.rdocumentation.org/packages/randomForest/versions/4.6-14/topics/randomForest

# Model summary
# Not super useful for model analysis
summary(wine_rf) 

# Objects returned from the model 
names(wine_rf)

# Predictions for test set
test_predictions_rf <- data.frame(testset,wine_rf$test$predicted)

# Accuracy for test set
mean(wine_rf$test$predicted==testset$quality_binary)

# Confusion matrix
table(wine_rf$test$predicted,testset$quality_binary)

# Quantitative measure of variable importance
importance(wine_rf)

# Sorted plot of importance
varImpPlot(wine_rf)

#cross validated error
train_test_split <- createDataPartition(wines$quality_binary,p=0.8,list=FALSE)

training_data <- wines[train_test_split,] #this is used to build the training model
test_data <- wines[-train_test_split,]
#10-fold cross validataion
cvSplits <- createFolds(training_data$quality_binary,k=10,)
#notice the number of elements in each fold
str(cvSplits)

#initialise accuracy vector
accuracies <- rep(NA,length(cvSplits))
i <- 0
#loop over all folds
for (testset_indices in cvSplits){
  i <- i+1
  trainset <- wines[-testset_indices, ]
  testset <- wines[testset_indices, ]
  w_rf <- randomForest(quality_binary ~.,data = trainset, 
                           importance=TRUE, xtest=testset[,-ncol(wines)],ntree=100)
  
  # Accuracy on test data
  accuracies[i] <- mean(w_rf$test$predicted==testset$quality_binary)
  
}

#a more unbiased error estimate (note the sd)
mean(accuracies)
sd(accuracies)

#Build final model on entire training data set (all folds)
wine_rf <- randomForest(quality_binary ~.,data = training_data, 
                         importance=TRUE, xtest=test_data[,-ncol(wines)],ntree=100)

#out of sample error
mean(wine_rf$test$predicted==test_data$quality_binary)

