# This file achives the tasks of analyzing a genetic data and building a classification model 
# for a specific gene. 
# This is done in two steps:
# 1) Reducing the dimensions of the data from 40,000 to few hundreds. Here we use Forward stage-wise regression
#    to iteratively eliminate irrelevant features. To achieve this the data is group into homogeneous buckets. 
# 2) Build a Logistic regression model with Lasso. The choice of L1 penalty also helps in dimension reduction as it 
#    can move parameters to zero. 
# In order to compensate for the limited RAM capacity and take advantage of parallel processing, we use the 
# 'Bigstep' package. 
# Finally, the predictions are saved in a csv file which gets uploaded in Kaggle. 
# This project achived a 97.7% accuracy. 




setwd("/Volumes/GoogleDrive/My Drive/Ph.D/Semester 2/Statistical Methods and Data Mining")
list.of.packages <- c("ggplot2","stats","gtools","bigstep","glmpathcr","xgboost","ROCR","sparsediscrim","LiblineaR","SparseM",
                      "biglasso","grplasso","covTest","gglasso","bigmemory",
                      "bigpca","ggplot2", "Rcpp","class","caret","pROC",
                      "glmnet","MASS","e1071","klaR","caret","mlbench",
                      "glm2","knitr","nnet","dplyr","ff","Metrics","leaps")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library(glmnet)
library(stats)
library(Metrics)
library(bigmemory)
library(bigpca)
library(bigstep)
library(ggplot2)
library(gtools)
set.seed(1234)

trainDF <- read.csv(file="Competition/train_dat.csv",header = FALSE,quote = "\"")
testDF <- read.csv(file="Competition/test_dat.csv",header = FALSE,quote = "\"")
train_class = read.csv(file="Competition/train_labels.csv",header = FALSE,quote = "\"")
names(train_class) <- 'outcome' 
data = cbind(train_class,trainDF)

outcome<-'outcome'



X <- as.big.matrix(trainDF,type="double")
y <- train_class
M <- ncol(X)
model <- selectModel(X, y, p=M, fitFun=fitLogistic,multif = TRUE)
model = mixedsort(model)
#model = selectModel(X, y, p=M, fitFun=fitLogistic,multif = FALSE)


pval = vector()
for(i in 1:length(model)){
n <- dim(trainDF)[1]
r = cor(trainDF[,model[i]],train_class)
df <- n - 2
t = sqrt(df) * r/sqrt(1 - r^2)
pval[i] = 2 * min(pt(t, df), pt(t, df, lower.tail = FALSE))
}
pval = t(rbind(model,pval))
pval = data.frame(pval,row.names = model)


cor_val = matrix(nrow = 20,ncol = 20)
cov_val = matrix(nrow = 20,ncol=20)
for(i in 1:length(model))
{
  for(j in 1:length(model))
cor_val[i,j] = var(trainDF[,model[i]],trainDF[,model[j]])
cov_val[i,j] = cov(trainDF[,model[i]],trainDF[,model[j]])
}

filtered_data = data.frame(V1=data[,model[1]])
XTest_filtered = data.frame(V1=testDF[,model[1]])
for(i in 2:length(model)){
  nextcol_data = data.frame(data[,model[i]])
  nextcol_test = data.frame(testDF[,model[i]])
  colnames(nextcol_data) <- c(paste("V", i,sep=""))
  colnames(nextcol_test) <- c(paste("V", i,sep=""))
  filtered_data = cbind(filtered_data,nextcol_data)
  XTest_filtered = cbind(XTest_filtered,nextcol_test)
}

filtered_data = cbind(train_class,filtered_data)
XTest_filtered = as.matrix(XTest_filtered)
error = vector()
for(i in 1:10){
  splitIndex <- createDataPartition(data[,outcome], p = .9, list = FALSE, times = 1)
  Train_split <- filtered_data[ splitIndex,]
  Val_split <- filtered_data[-splitIndex,]

  YTrain = Train_split[,outcome]
  XTrain_filtered = Train_split[,-1]
  YVal = Val_split[,outcome]
  XVal_filtered= Val_split[,-1]
  
  XTrain_filtered = as.matrix(XTrain_filtered)
  XVal_filtered = as.matrix(XVal_filtered)
  YTrain <- as.factor(as.matrix(YTrain))
  YVal <- as.factor(as.matrix(YVal))
  XTest_filtered = as.matrix(XTest_filtered)
  
cv.lasso <- cv.glmnet(XTrain_filtered, YTrain, nlambda=100,family='binomial',alpha = 1,nfolds=10)
predTrain = predict(cv.lasso,XVal_filtered,s = "lambda.min",type='class')

error[i] = ce(YVal,predTrain)
}
print(error)


cv.lasso <- cv.glmnet(as.matrix(filtered_data[,-1]), filtered_data[,1], nlambda=100,family='binomial',alpha = 1,nfolds=10)
predTrain = predict(cv.lasso,XTest_filtered,s = "lambda.min",type='class')
pred = data.frame(id = paste0("n", 1:1200), prediction = paste0(predTrain))
write.csv(pred, "prediction.csv",row.names=FALSE)



