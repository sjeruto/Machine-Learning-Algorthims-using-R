#### : GBM ####

# This exercise involves building a gbm model 
rm(list=ls())
library(gbm)
library(parallel)

#Here we will use a dataset from the UCI repository. It contains various chemical properties about red and white wines and their quality score (1-10). The original data has two sets (for red and white) however I have concatenated these together and simply added a column for whether it is red or white. I have also truncated the quality to being only 'good' or 'bad' where good is >7 and bad is under

# Read in and format our data
wine_data <- read.csv("wines.csv")
str(wine_data)
#convert colour to factor
wine_data$colour <- as.factor(wine_data$colour)

# Drop the ID column
wine_data$ID <- NULL

# Let's make this a binary classification
wine_data$quality_binary <- 1
wine_data[wine_data$quality < 7, "quality_binary"] <- 0
# wine_data$quality_binary <- as.factor(wine_data$quality_binary)

# Need to drop the quality predictor
wine_data$quality <- NULL

# Our dataset is a littl unbalance but not too bad.
table_1 <- table(wine_data$quality_binary)
table_1
prop.table(table_1) 

# Create training and test sets. This process should be familiar by now
trainset_size <- floor(0.80 * nrow(wine_data))
set.seed(1) 
trainset_indices <- sample(seq_len(nrow(wine_data)), size = trainset_size)
training <- wine_data[trainset_indices, ]
testset <- wine_data[-trainset_indices, ]

# Checks
nrow(training)
nrow(testset)
nrow(wine_data)

#### Train the model ####

# Defining some parameters

gbm_depth = 2 #maximum nodes per tree
gbm_n_min = 20 #minimum number of observations in the trees terminal, important effect on overfitting
gbm_shrinkage=0.01 #learning rate
cores_num = detectCores() - 1 #number of cores. Leave 1 for OS. Beware this setting!
gbm_cv_folds=5 #number of cross-validation folds to perform
num_trees = 200 # Number of iterations

start <- proc.time()

# fit initial model
gbm_clf = gbm(training$quality_binary~.,
                  data=training[, -ncol(training)],
                  distribution='bernoulli', #binary response
                  n.trees=num_trees,
                  interaction.depth= gbm_depth,
                  n.minobsinnode = gbm_n_min, 
                  shrinkage=gbm_shrinkage, 
                  cv.folds=gbm_cv_folds,
                  verbose = TRUE, #print the preliminary output
                  n.cores = cores_num
)

end <- proc.time() - start
end_time <- as.numeric((paste(end[3])))
end_time

# Estimate the optimal number of iterations (when will the model stop improving)
# The black is the training deviance dropping whilst the green is the test.
best_iter = gbm.perf(gbm_clf, method = "cv")
print(best_iter)

# Gives the variable importance in a graph
summary(gbm_clf,n.trees=best_iter, ylab = "Variable", main = "Variable Relative Importance")

# OR just as a table
summary(gbm_clf)

# Let us get our estimates
testset$probability = predict(gbm_clf, testset, n.trees = best_iter, type = "response")
testset$prediction = 0

# Modify the probability threshold to see if you can get a better accuracy
testset[testset$probability >= 0.5, "prediction"] = 1

# Confusion matrix
confusion_matrix <- table(pred=testset$prediction,true=testset$quality_binary)
confusion_matrix

# Accuracy
mean(testset$prediction==testset$quality_binary)

#### EXERCISES s####

# Take a look at the deviance plot - 
#Notice how the testing error bottoms out but training keeps getting better? What does this indicate?

#1. Which hyperparameters will you change? Try tuning some and seeing how that affects things
#2. Split the dataset into red and white wines.
#Now run a new gbm model on each. Is it better? How does the feature importances change? 
#What does this mean?

# Can you improve the accuracy on the test set?

