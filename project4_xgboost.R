library(ggplot2)
library(car)
library(dplyr)
library(corrplot)
library(caret)
library(xgboost)
set.seed(0)
data<-read.csv("train.csv")
kag.test<-read.csv("test.csv")[,-1]
data.full<-rbind(data[,-c(1,132)],kag.test)


###split into two class: catergory, quantitative
#data.all.cat<-select(data.full,contains("cat"))
#data.all.qua<-select(data.full,contains("con"))

###dummy all varable
dm.all.train<-as.data.frame(predict(preProc,model.matrix(~.,data.full)[1:188318,]))
#dm.all.cat<-model.matrix(~.,data.all.cat)
#dm.train.cat<-model.matrix(~.,data.all.cat)[1:188318,]
dm.all.kagg<-as.data.frame(predict(preProc,model.matrix(~.,data.full)[188319:313864,]))
#dm.test.cat<-model.matrix(~.,data.all.cat)[188319:313864,]

#dm.train.cat<-as.data.frame(predict(preProc,dm.train.cat))

#dm.test.cat<-as.data.frame(predict(preProc,dm.test.cat))

### normalize quantitive 
normalize = function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}
#data.all.qua.nom<-as.data.frame(lapply(data.all.qua,normalize))

loss.all.nom<-normalize(log(data$loss+1))
##summary(loss.all.nom)
##   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##0.0000  0.5881  0.6384  0.6410  0.6923  1.0000 
loss.train<-loss.all.nom

## catergory subset data frame

dm.all.train$loss<-loss.all.nom
set.seed(0)
train<-sample(1:nrow(dm.all.train),0.7*nrow(dm.all.train))
model.xgboost<-xgboost(data=as.matrix(dm.all.train[train,1:1037]),
                       label = dm.all.train$loss[train],
                       max.depth=10,
                       objective="reg:linear",
                       eta=1,
                       cv=5,
                       eval_metric = "error",
                       early.stop.round = 10,
                       nrounds = 520)
score<-c()
n.turn<-c()
for (i in 1:100){
 model.xgboost.cv<-xgboost(data=as.matrix(dm.all.train[train,1:1037]),
                         label = dm.all.train$loss[train],
                         max.depth=9,
                         objective="reg:linear",
                         eta=1,
                         cv=5,
                         eval_metric = "error",
                         early.stop.round = 10,
                         nrounds = 3000)
 score[i]=model.xgboost.cv$bestScore
 n.turn[i]=model.xgboost.cv$bestInd
}

index=which(score==min(score))
n.turn[index]
n.turn[10]  ##depth10 ,ntrun=520

eta=10^(seq(-5, .5, length = 100))
score_2=c()
n.turn_2=c()
for (j in 1:100){
  model.xgboost.cv_2<-xgboost(data=as.matrix(dm.all.train[train,1:1037]),
                            label = dm.all.train$loss[train],
                            max.depth=9,
                            objective="reg:linear",
                            eta=eta[j],
                            cv=5,
                            eval_metric = "error",
                            early.stop.round = 50,
                            nrounds = 520)
  score_2[j]=model.xgboost.cv_2$bestScore
  n.turn_2[j]=model.xgboost.cv_2$bestInd
}
index_2=which(score_2==min(score_2))   ###94 is best eta[94]=1.467799, n.turn_2[94]=517
n.turn_2[index_2]
ggplot()+geom_line(aes(x=eta,y=score_2,color="Error"))+theme_bw()

score_3<-c()
n.turn_3<-c()

for (k in 4:15){
  model.xgboost.cv_3<-xgboost(data=as.matrix(dm.all.train[train,1:1037]),
                              label = dm.all.train$loss[train],
                              max.depth=k,
                              objective="reg:linear",
                              eta=1.467799,
                              cv=5,
                              eval_metric = "error",
                              early.stop.round = 50,
                              nrounds = 600)
  score_3[k-3]=model.xgboost.cv_3$bestScore
  n.turn_3[k-3]=model.xgboost.cv_3$bestInd
}
index_3=which(score_3==min(score_3))   ###max depth=10  n.turn_3[7]=560
n.turn_3[index_3]
ggplot()+geom_line(aes(x=4:15,y=score_3,color="Error"))+theme_bw()



###best parameters
model.xgboost.best<-xgboost(data=as.matrix(dm.all.train[train,1:1037]),
                            label = dm.all.train$loss[train],
                            max.depth=13,
                            objective="reg:linear",
                            eta=1.467799,
#                            cv=5,
                            eval_metric = "error",
                            early.stop.round = 50,
                            nrounds = 223)
a=max(log(data$loss+1)) ###11.70365532
b=min(log(data$loss+1))  ###0.5128236264
kaggle.pre<-predict(model.xgboost.best,as.matrix(dm.all.kagg))
result<-exp(kaggle.pre*(a-b)+b)-1
write.csv(result,"xgboost_all_2.csv")

param <- list(max_depth=2, eta=1, silent=1, nthread=2, objective='reg:linear')
nround=seq(10,1000,by=10)
dtrain <- xgb.DMatrix(as.matrix(dm.all.train[train,1:1037]), label =dm.all.train$loss[train])
cv<-xgb.cv(param, dtrain, nround, nfold=5, metrics={'error'})

pre.test<-predict(model.xgboost,as.matrix(dm.all.train[-train,1:1037]))

###recreate the dataframe with loss
# train.qua<-mutate(data.all.qua.nom[1:188318,],loss=loss.train)
# test.qua<-data.all.qua.nom[188319:313864,]

# formula<-as.formula(paste("loss~",paste(names(data.all.qua.nom),collapse = "+")))


### neural network to create the nonlinear term
# library(neuralnet)
# set.seed(0)
# train<-sample(1:nrow(train.qua),0.7*nrow(train.qua))
# model.qua<-neuralnet(formula = formula,
#                      hidden = 1,
#                      data = train.qua[train,])


# set.seed(0)
# train<-sample(1:nrow(dm.train.cat),0.7*nrow(dm.train.cat))
# model.boost<-gbm(loss ~ ., data = dm.train.cat[train, ],
#                  distribution = "gaussian",
#                  n.trees = 2000,
#                  interaction.depth = 6,
#                  shrinkage = 0.01)
# save(model.boost,file = "./model.boost.cat.rda")
# ntrees.tune<-seq(from=100,to=2000,by=100)
# p<-predict(model.boost,n.trees = ntrees.tune,newdata = dm.train.cat[-train,])
# berr2 = with(dm.train.cat[-train, ], apply((p - loss)^2, 2, mean))
# a=max(log(data$loss+1)) ###11.70365532
# b=min(log(data$loss+1))  ###0.5128236264
# y.htrain<-model.boost$fit
# train.pre<-exp(y.htrain*(a-b)+b)-1
# 
# write.csv(train.pre,"Cat_only_trainValue.csv")


# y.htest<- exp(p[,20]*(a-b)+b)-1
# sum((y.htest-data$loss[-train])^2)/nrow(p)
# sum((log(y.htest)-log(data$loss[-train]))^2)/nrow(p)
# 
# par(mfrow = c(1, 1))
# plot(ntrees.tune, as.vector(berr2), pch = 16,
#      ylab = "Mean Squared Error",
#      xlab = "# Trees",
#      main = "Boosting Test Error")
# library(ggplot2)
# ggplot()+geom_point(aes(x=ntrees.tune,y=as.vector(berr2)))



####kagle test
# kaggle.pre<-predict(model.boost,newdata = dm.test.cat, n.trees=2000)
# result<-exp(kaggle.pre*(a-b)+b)-1
# write.csv(result,"catergoryonly.csv")
# 
# 
# 
# ######
# set.seed(0)
# train<-sample(1:nrow(dm.train.cat),0.7*nrow(dm.train.cat))
# model.boost.2<-gbm(loss ~ ., data = dm.train.cat[train, ],
#                    distribution = "gaussian",
#                    n.trees = 3000,
#                    interaction.depth = 6,
#                    shrinkage = 0.01)