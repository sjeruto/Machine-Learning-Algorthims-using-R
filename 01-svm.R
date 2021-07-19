#################################################
##PART1 - generate a linearly separable dataset
################################################


#number of datapoints
n <- 200
#seed
set.seed(42)
#Generate dataframe with 2 uniformly distributed predictors lying between 0 and 1.
#Call the 2 predictors are x1 and x2

df <- data.frame(x1=runif(n),x2=runif(n))


# lets make the line x1 = x2 the separation boundary and set the class (y) to be -1 and 1
# for points that lie below and above the line respectively.

df$y <- factor(ifelse(df$x1-df$x2>0,-1,1),levels=c(-1,1))


#plot data with separation line

# remember y = mx + c is the equation of a straight line, here x = x1 and y = x2, so m = 1and c=0
#
#differentiate class by colour
#suppress legend

#load ggplot
library(ggplot2)
#build plot
p <- ggplot(data=df, aes(x=x1,y=x2,colour=y)) + 
  geom_point() +  scale_colour_manual(values=c("-1"="red","1"="blue")) + 
  geom_abline(slope=1,intercept=0)
#display it  
p


#The separator has no margin, let's give it one....
#To do this, we need to exclude points that lie close to the decision boundary. 
#One way to do this is to exclude points that have a x1 and x2 values that differ by less
#than a specified value, delta. Let's set delta to 0.05...

#create data set with margin
#this will be a subset of the previous data set

#create a margin of 0.05 in dataset
delta <- 0.05
# retain only those points that lie outside the margin
df1 <- df[abs(df$x1-df$x2)>delta,]
#check number of datapoints remaining
nrow(df1)  

#replot dataset with margin (code is exactly same as before)
p <- ggplot(data=df1, aes(x=x1,y=x2,colour=y)) + 
  geom_point() + scale_colour_manual(values=c("red","blue")) + 
  geom_abline(slope=1,intercept=0)
p

#There is a clear empty space on either side of the boundary. Let's add in boundary line on either side.
#These should be offset from the decision boundary by delta (one on either side). The lines that
#define these, will have the same slope as the decision boundary but will have a y intercept of delta
#and -delta respectively

p <- p + 
  geom_abline(slope=1,intercept = delta, linetype="dashed") +
  geom_abline(slope=1,intercept = -delta, linetype="dashed")
p

################################################
##PART2 - build linear svm
################################################

#split into train and test
set.seed(1)

## 80% of the sample size, use floor to round down to nearest integer
trainset_size <- floor(0.80 * nrow(df1))

trainset_indices <- sample(seq_len(nrow(df1)), size = trainset_size)

#assign observations to training and testing sets

trainset <- df1[trainset_indices, ]
testset <- df1[-trainset_indices, ]

#rowcounts to check
nrow(trainset)
nrow(testset)
nrow(df1)


write.csv(trainset,file="linear_svm_trainset.csv",row.names = FALSE)
write.csv(testset,file="linear_svm_testset.csv",row.names = FALSE)


#the svm  classifier in e1071 has a number of parameters, we'll look only at key parameters:
# ~ - formula specifying classification and predictor variables
#data - dataframe containing dataset
#type - classification method to be used. We will use C-classification (see the docs for others)
#kernel - this is essentially the type of function describing the decision boundary. This set to "linear" as the data 
#is linearly separable
#cost and gamma - these are parameters that are used to tune the model. \
#Note that gamma has no influence on linear kernels. Cost is essentially a penalty for
#margin violations. The lower the cost, the greater the number of margin violations.
#We will use default value for cost C (=1) to begin with and see the effect of changing it later
#scale - svm scales variables by default. Suppress scaling to allow comparison to original data
#In real life, you will typically set scaling=TRUE to account for large differences in
#the magnitudes of predictors.

library(e1071)

svm_model<- 
  svm(y ~ ., data=trainset, type="C-classification", kernel="linear", scale=FALSE)


svm_model


#Note the large number of support vectors, this is because we have set a relatively low cost for margin violation. This manifests
#itself as a large number of datapoints lying within the margins.


#training accuracy
pred_train <- predict(svm_model,trainset)
mean(pred_train==trainset$y)

#test accuracy
pred_test <- predict(svm_model,testset)
mean(pred_test==testset$y)

################################################
##PART 3 - Explore model object
################################################


#key contents of svm object

#index of support vectors in training dataset
svm_model$index

#Support vectors
svm_model$SV

#negative intercept (unweighted)
svm_model$rho

#Weighting coefficients for support vectors
svm_model$coefs

################################################
#PART 4 -  Visualise the classifier using ggplot
################################################

#visualise training data, distinguish classes using colour

p <- ggplot(data=trainset, aes(x=x1,y=x2,colour=y)) + 
  geom_point()+ scale_colour_manual(values=c("red","blue"))
p
#identify support vectors 
df_sv <- trainset[svm_model$index,]

#mark out support vectors in plot
p <- p + geom_point(data=df_sv,aes(x=x1,y=x2),colour="purple",size = 4,alpha=0.5)
p

#Explanation- we first plot all points in the training set. Then we overlay a purple circle
#on support vectors.

#We know the decision boundary is a straight line (why?)

#OK, so the next step is to extract the slope and intercept of the straight line from the
#svm object. This, unfortunately, needs some work...

#We use coefs and the support vectors to build the what's called the weight vector.
#The weight vector is the product of the coefs matrix with the matrix containing the SVs.
#Note, it makes sense that only the SVs play a role in defining the decision boundary. Why?

#build weight vector
w <- t(svm_model$coefs) %*% svm_model$SV #weight vector


#slope = -w[1]/w[2], intercept = rho/w[2]. Note that the intercept <> 0 and the slope
#is slightly less than 1. We know this is incorrect since we have designed the dataset
#to have a boundary of slope 1 and intercept 0. We'll see how to do better shortly

#calculate slope and save it to a variable
slope_1 <- -w[1]/w[2]

#calculate intercept and save it to a variable
intercept_1 <- svm_model$rho/w[2]

#plot decision boundary based on  calculated slope and intercept
p <- p + geom_abline(slope=slope_1,intercept = intercept_1)

p

#Notice that the boundary is "supported" by roughly the same number of support vectors
#on either side. This makes intuitive sense.

#Margins have the same slope (parallel to decision boundary) and lie +-1/w[2] on either
#side of the boundary

#add margins to plot
p <- p + 
  geom_abline(slope=slope_1,intercept = intercept_1-1/w[2], linetype="dashed")+
  geom_abline(slope=slope_1,intercept = intercept_1+1/w[2], linetype="dashed")

#display plot
p

#Notice that the support vectors lie within the margins. That is, we have allowed margin
#violations. Classifiers that allow margin violations are called soft margin classifiers.

#can also plot using function provided in e1071
#formula not needed since this is a 2 variable problem

plot(x=svm_model, data=trainset)
#Note that axes for x1 and x2 have been interchanged. Support vectors are marked as x.





