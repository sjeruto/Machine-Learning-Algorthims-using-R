#### Exercise 2: Data Partitioning ####
rm(list=ls())
# Some packages
library (rpart)
library(mlbench)
library(caret)

# Load Ionosphere  dataset (classification problem)
data("Ionosphere")
class_col_num <- grep("Class",names(Ionosphere))


trainset_indices <- createDataPartition(Ionosphere$Class,p=0.8)
#createDataPartition creates _stratified_ random splits

#create n partitions of the same dataset (similar to what we did manually in the
#multiple runs function)

trainset_index_list <- createDataPartition(Ionosphere$Class,p=0.8, times=10)

trainset_index_list

#reimplement multiple runs using createDataPartition

accuracies <- rep(NA,length(trainset_index_list))
i <- 0
for (trainset_indices in trainset_index_list){
  i <- i+1
  trainset <- Ionosphere[trainset_indices, ]
  testset <- Ionosphere[-trainset_indices, ]
  rpart_model <- rpart(Class~.,data = trainset, method="class")
  # Predict on test data
  rpart_predict <- predict(rpart_model,testset[,-class_col_num],type="class")
  # Accuracy
  accuracies[i] <- mean(rpart_predict==testset$Class)
  
}

mean(accuracies)
sd(accuracies)

#complete model building process using cross validation

#Step 1: create holdout (test) set
train_test_split <- createDataPartition(Ionosphere$Class,p=0.8,list=FALSE)

training_data <- Ionosphere[train_test_split,] #this is used to build the training model
test_data <- Ionosphere[-train_test_split,] #this is used to test the final model only

#10-fold cross validataion
cvSplits <- createFolds(training_data$Class,k=10,)
#notice the number of elements in each fold
str(cvSplits)

#initialise accuracy vector
accuracies <- rep(NA,length(cvSplits))
i <- 0
#loop over all folds
for (testset_indices in cvSplits){
  i <- i+1
  trainset <- Ionosphere[-testset_indices, ]
  testset <- Ionosphere[testset_indices, ]
  rpart_model <- rpart(Class~.,data = trainset, method="class")
  # Predict on test data
  rpart_predict <- predict(rpart_model,testset[,-class_col_num],type="class")
  # Accuracy
  accuracies[i] <- mean(rpart_predict==testset$Class)
  
}

#a more unbiased error estimate (note the sd)
mean(accuracies)
sd(accuracies)

#Build final model on entire training data set (all folds)
rpart_model <- rpart(Class~.,data = training_data, method="class")

#predict on test data
rpart_predict <- predict(rpart_model,test_data[,-class_col_num],type="class")

#out of sample error (error on test set - note this data is not touched during model building)
mean(rpart_predict==test_data$Class)

