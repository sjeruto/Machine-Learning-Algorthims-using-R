#### Exercise 3: Pruning ####

# Clear everything
rm(list=ls())

# Load rpart, rpart.plot and mlbench if you haven't done so
library (rpart)
library(rpart.plot)
library(mlbench)


# Load Sonar  dataset (classification problem)
data(Sonar)
# https://www.rdocumentation.org/packages/mlbench/versions/2.1-1/topics/Sonar

# Variable names
names(Sonar)

# Get index of predicted variable
class_col_num <- grep("Class",names(Sonar))

#### Training testing set ####

# Create training and test sets
## 80% of the sample size, use floor to round down to nearest integer
trainset_size <- floor(0.80 * nrow(Sonar))


# First step is to set a random seed to ensure we get the same result each time
set.seed(1) 

# Get indices of observations to be assigned to training set. 
# This is via randomly picking observations using the sample function

trainset_indices <- sample(seq_len(nrow(Sonar)), size = trainset_size)

# Assign observations to training and testing sets

trainset <- Sonar[trainset_indices, ]
testset <- Sonar[-trainset_indices, ]

# Rowcounts to check
nrow(trainset)
nrow(testset)
nrow(Sonar)
# Are training and testsets representative?
table(trainset$Class)
table(testset$Class)


#### Build a basic tree ####

# Default params. This is a classification problem so set method="class"
rpart_model <- rpart(Class~.,data = trainset, method="class")


# Plot tree (UNPRUNED)
prp(rpart_model)


# Unpruned model 
rpart_predict <- predict(rpart_model,testset[,-class_col_num],type="class")
mean(rpart_predict==testset$Class)

# Confusion matrix 
table(pred=rpart_predict,true=testset$Class)

#### Part 2: Cost complexity pruning ####

# Cost-complexity plot - can you see the minimum in the plot?
plotcp(rpart_model)

# https://www.rdocumentation.org/packages/rpart/versions/4.1-13/topics/plotcp

# What's the tree size which results in the min cross validated error
optmin <- which.min(rpart_model$cptable[,"xerror"])
optmin
# min cross validated error can give trees that are too deep. 
# empirically, the smallest tree size within 1 sd of the min results in better generalizability
cpminerr <- rpart_model$cptable[optmin, "xerror"]
opt1se <- which.min(abs(rpart_model$cptable[,"xerror"]-cpminerr-cpminse))
opt1se
cpminse <- rpart_model$cptable[optmin, "xstd"]

# Value of the complexity parameter (alpha) for that gives a a tree of optmin or opt1se
cpmin <- rpart_model$cptable[optmin, "CP"]
cp1se <- rpart_model$cptable[opt1se, "CP"]

# "prune" the tree using that value of the complexity parameter (try both cpmin and cp1se)
pruned_model <- prune(rpart_model,cpmin) 

# Plot pruned tree
prp(pruned_model)

# Predictions from pruned model
rpart_pruned_predict <- predict(pruned_model,testset[,-class_col_num],type="class")

# Accuracy of pruned model
mean(rpart_pruned_predict==testset$Class)

# Confusion matrix (PRUNED model)
table(pred=rpart_pruned_predict,true=testset$Class)

