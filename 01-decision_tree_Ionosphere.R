
# Upload libraries
library (rpart)
library(rpart.plot)
library(mlbench)


# Load Ionosphere  dataset (classification problem)
data("Ionosphere")


# Explore dataset
nrow(Ionosphere)
ncol(Ionosphere)
summary(Ionosphere)

# Find name of predicted variable
names(Ionosphere)
# Get index of predicted variable
class_col_num <- grep("Class",names(Ionosphere))

#### Test Train Split ####

# Create training and test sets
## 80% of the sample size, use floor to round down to nearest integer
trainset_size <- floor(0.80 * nrow(Ionosphere))

# First step is to set a random seed to ensure we get the same result each time
set.seed(42) 

# Get indices of observations to be assigned to training set.
# This is via randomly picking observations using the sample function

trainset_indices <- sample(seq_len(nrow(Ionosphere)), size = trainset_size)

# Assign observations to training and testing sets

trainset <- Ionosphere[trainset_indices, ]
testset <- Ionosphere[-trainset_indices, ]

# Rowcounts to check
nrow(trainset)
nrow(testset)
nrow(Ionosphere)


#### Build a tree ####

# Default params. This is a classification problem so set method="class"
rpart_model <- rpart(Class~.,data = trainset, method="class")

# Plot tree - SAVE PLOT for comparison later
plot(rpart_model);text(rpart_model)

# prp from rpart.plot produces nicer plots
prp(rpart_model)

# Summary
summary(rpart_model)

# Try out an even prettier graph. 
# Customise as per https://www.rdocumentation.org/packages/rpart.plot/versions/3.0.6/topics/rpart.plot
rpart.plot(rpart_model, # middle graph
           type=2,
           extra=101, 
           box.palette="GnBu",
           shadow.col="gray"
)

# Predict on test data
rpart_predict <- predict(rpart_model,testset[,-class_col_num],type="class")


# Accuracy
mean(rpart_predict==testset$Class)

#confusion matrix
cfm <- table(pred=rpart_predict,true=testset$Class)
cfm

#Precision = TP/(TP+FP)
precision <- cfm[1,1]/(cfm[1,1]+cfm[1,2])
precision

#Recall = TP/(TP+FN)
recall <- cfm[1,1]/(cfm[1,1]+cfm[2,1])
recall

#F1
f1 <- 2*(precision*recall/(precision+recall))
f1

#### Part 2: Different seeds ####

# Now rerun for different values of seed (say 53 and 1)  and re-plot tree, 
#calculate confusion matrix and performance measures for each. Comment on your results.

#### Part 3: Average across multiple partitions ####

# A simple way to get more reliable performance measures is to calculate average measures over different data partitions.To do this, we'll write a function to build multiple rpart models over different partitions and return a vector of calculated accuracies for all models.

#Note: the following function loops over code discussed earlier, using a different data partion in each iteration

multiple_runs_rpart <-function(df,class_variable_name,train_fraction,nruns)
multiple_runs_rpart 
  
#Purpose:
  #Builds rpart model for nrun data partitions
  
  #Return value:
  #Vector containing nrun accuracies
  
  #Arguments:
  #df: variable containing dataframe
  #class_variable_name: class name as a quoted string. e.g. "Class"
  #train_fraction: fraction of data to be assigned to training set (0<train_fraction<1)
  #nruns: number of data partitions
  
  # Find column index of class variable
  class_col_num <- grep(class_variable_name,names(df))
  # Initialize accuracy vector
  accuracies <- rep(NA,nruns)
  # Set seed (can be any integer)
  set.seed(42)
  for (i in 1:nruns){
    # Partition data
    trainset_size <- floor(train_fraction * nrow(df))
    trainset_indices <- sample(seq_len(nrow(df)), size = trainset_size)
    trainset <- df[trainset_indices, ]
    testset <- df[-trainset_indices, ]
    # Bbuild model 
    # Paste builds formula string and as.formula interprets it as an R formula
    rpart_model <- rpart(as.formula(paste(class_variable_name,"~.")),data = trainset, method="class")
    # Predict on test data
    rpart_predict <- predict(rpart_model,testset[,-class_col_num],type="class")
    # Accuracy
    accuracies[i] <- mean(rpart_predict==testset[[class_variable_name]])
  }
  return(accuracies)
}

# Calculate average accuracy and std dev over 30 random partitions
accuracy_results <- multiple_runs_rpart(Ionosphere,"Class",0.8,30)
mean(accuracy_results)
sd(accuracy_results)
accuracy_results
