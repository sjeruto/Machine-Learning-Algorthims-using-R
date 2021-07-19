rm(list=ls())
library(gbm)
library(parallel)

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


# create grid search
hyper_grid <- expand.grid(
  learning_rate = c(0.05, 0.01,0.005),
  minobs = c(10,5,3),
  depth= c(2,5,7),
  error = NA,
  trees = NA
)

# execute grid search
for(i in seq_len(nrow(hyper_grid))) {
  
  # fit gbm
  set.seed(123)  # for reproducibility
  train_time <- system.time({
    m <- gbm(
      training$quality_binary~.,
      data=training[, -ncol(training)],
      distribution='bernoulli', #binary response
      n.trees = 1000, 
      shrinkage = hyper_grid$learning_rate[i], 
      interaction.depth = hyper_grid$depth[i], 
      n.minobsinnode = hyper_grid$minobs[i],
      cv.folds = 10,
      verbose = TRUE, #print the preliminary output
      n.cores = detectCores() - 1
    )
  })
  
  # add SSE, trees, and training time to results
  hyper_grid$error[i]  <- sqrt(min(m$cv.error))
  hyper_grid$trees[i] <- which.min(m$cv.error)
  hyper_grid$Time[i]  <- train_time[["elapsed"]]
  
}

hyper_grid[order(hyper_grid$error),]
