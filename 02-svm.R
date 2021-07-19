#clear environment
rm(list=ls())
dev.off()

#load data partitioned earlier
trainset <- read.csv(file="linear_svm_trainset.csv")
trainset$y <- as.factor(trainset$y)
testset <- read.csv(file="linear_svm_testset.csv")
testset$y <- as.factor(testset$y)

library(e1071)

# run the code from here to the end for different
#values of cost (say 10, 100, 1000) and compare to default cost case
#in previous exercise. In particular, the number of suport vectors and
# the margin width

svm_model<- 
  svm(y ~ ., data=trainset, type="C-classification", kernel="linear", cost=100, scale=FALSE)

#print summary - comment on the number of support vectors
summary(svm_model)

#training accuracy
pred_train <- predict(svm_model,trainset)
mean(pred_train==trainset$y)

#test accuracy
pred_test <- predict(svm_model,testset)
mean(pred_test==testset$y)

#Visualise using ggplot

#visualise training data, distinguish classes using colour

p1 <- ggplot(data=trainset, aes(x=x1,y=x2,colour=y)) + 
  geom_point()+ scale_colour_manual(values=c("red","blue"))

#identify support vectors 
df_sv <- trainset[svm_model$index,]

#mark out support vectors in plot
p1 <- p1 + geom_point(data=df_sv,aes(x=x1,y=x2),colour="purple",size = 4,alpha=0.5)
p1



#build weight vector
w <- t(svm_model$coefs) %*% svm_model$SV #weight vector


#calculate slope and save it to a variable
slope_1 <- -w[1]/w[2]

#calculate intercept and save it to a variable
intercept_1 <- svm_model$rho/w[2]

#plot decision boundary based on  calculated slope and intercept
p1 <- p1 + geom_abline(slope=slope_1,intercept = intercept_1)


#Margins have the same slope (parallel to decision boundary) and lie +-1/w[2] on either
#side of the boundary

#add margins to plot
p1 <- p1 + 
  geom_abline(slope=slope_1,intercept = intercept_1-1/w[2], linetype="dashed")+
  geom_abline(slope=slope_1,intercept = intercept_1+1/w[2], linetype="dashed")

#display plot
p1

#try other values of cost 100, 1000 and comment on the margin
