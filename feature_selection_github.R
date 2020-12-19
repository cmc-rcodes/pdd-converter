# libraries needed
library(glmnet)
library(caret)
library(dplyr)
library(e1071)

discovery <- data.frame(discovery)


# custom control parameters for 10 fold cross validation
custom <- trainControl(method="repeatedcv",
                       number=10,
                       repeats=10,
                       verboseIter = T)

# feature selection step in 10000 loops
features <-list()
permnum <- 10000
for (i in 1:permnum){
 
   ## data split in training and test sets with 3:1 ratio maintaining balance in converter numbers
  set.seed(i)
  trainIndex<- createDataPartition(discovery$converter, p=0.75, list=FALSE)
  train <-discovery[trainIndex,]
  test <-discovery[-trainIndex,]
  
  ## training
  set.seed(i)
  lasso <- train(converter ~ .,
                 train,
                 method = 'glmnet',
                 tuneGrid=expand.grid(alpha=1,
                                      lambda=seq(0.0001,0.05,length=5)), 
                 trControl=custom)
  
  ## extract features for the best performing model
  feature <- data.frame(feature.name = dimnames(coef(lasso$finalModel,s=lasso$bestTune$lambda))[[1]], 
                     feature.coef = matrix(coef(lasso$finalModel,s=lasso$bestTune$lambda)))
  
  ## exclude the (Intercept) term
  feature <- feature[-1,]
  
  ## sort coefficients in ascending order
  feature <- dplyr::arrange(feature,-feature.coef)
  
  ## select features with coefficients > 0
  selected_feature <- dplyr::filter(feature,feature.coef!=0) 
  
  features[[i]] <-selected_feature$feature.name
}

# varImp plot
lasso
plot(varImp(lasso,scale=F))
features.name <- unlist(features)

# features selected >= 5000 times
features.name.df <- as.data.frame(table(features.name))
ordered_features.name.df <- dplyr::arrange(features.name.df,-Freq)
selected_500_features <- subset(ordered_features.name.df, Freq >= 5000)
features.var <- unlist(selected_500_features$features.name)


