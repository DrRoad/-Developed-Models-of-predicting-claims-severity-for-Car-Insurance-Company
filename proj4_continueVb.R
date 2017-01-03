library(ggplot2)
library(car)
library(dplyr)
library(corrplot)
library(caret)
library(gbm)
set.seed(0)
data<-read.csv("train.csv")
kag.test<-read.csv("test.csv")[,-1]
data.full<-rbind(data[,-c(1,132)],kag.test)


###split into two class: catergory, quantitative
data.all.cat<-select(data.full,contains("cat"))
#data.all.qua<-select(data.full,contains("con"))

###dummy catergory class
dm.all.cat<-model.matrix(~.,data.all.cat)
dm.train.cat<-model.matrix(~.,data.all.cat)[1:188318,]
dm.test.cat<-dm.all.cat[188319:313864,]

dm.train.cat<-as.data.frame(predict(preProc,dm.train.cat))

### normalize quantitive 
normalize = function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}
data.all.qua.nom<-as.data.frame(lapply(data.all.qua,normalize))

loss.all.nom<-normalize(log(data$loss+1))
dm.train.cat$loss<-loss.all.nom
##summary(loss.all.nom)
##   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##0.0000  0.5881  0.6384  0.6410  0.6923  1.0000 
loss.train<-loss.all.nom

###recreate the dataframe with loss
train.qua<-mutate(data.all.qua.nom[1:188318,],loss=loss.train)
test.qua<-data.all.qua.nom[188319:313864,]

formula<-as.formula(paste("loss~",paste(names(data.all.qua.nom),collapse = "+")))


### neural network to create the nonlinear term
library(neuralnet)
set.seed(0)
train<-sample(1:nrow(train.qua),0.7*nrow(train.qua))
model.qua<-neuralnet(formula = formula,
                     hidden = 10,
                     data = train.qua[train,])


set.seed(0)
model.boost<-gbm(loss ~ ., data = Boston[train, ],
                 distribution = "gaussian",
                 n.trees = 2000,
                 interaction.depth = 6,
                 shrinkage = 0.01))
library(h2o)
h2o.init(nthreads = -1)
train.h2o<-as.h2o(train.qua[train,])

kag.h2o<-as.h2o(test.qua)

set.seed(0)

model.con<-h2o.deeplearning(x=colnames(train.h2o[1:14]),
                        y="loss",
                        training_frame = train.h2o,
                        activation = "TanhWithDropout",
                        input_dropout_ratio = 0.2, # % of inputs dropout
                        hidden_dropout_ratios = c(0.5,0.5), # % for nodes dropout
#                        balance_classes = TRUE, 
                        hidden = c(300,100), # three layers of 50 nodes
                        epochs = 500,# max. no. of epochs
#                      l2 = 0.01,
                        nfolds = 10)
summary(model.con)
model
plot(model)
a=max(log(data$loss+1)) ###11.70365532
b=min(log(data$loss+1))  ###0.5128236264

yhat_train <- as.data.frame(h2o.predict(model.con, train.h2o)$predict)
yhat_train<-exp(yhat_train*(a-b)+b)-1
  
write.csv(yhat_train,"con_train2.csv")

sum((yhat_train-as.data.frame(train.h2o$loss)$loss)^2)/nrow.H2OFrame(train.h2o)


yhat_test <- as.data.frame(h2o.predict(model, test.h2o)$predict)

sum((yhat_test-as.data.frame(test.h2o$loss)$loss)^2)/nrow.H2OFrame(test.h2o)


result.kag.con<-as.data.frame(h2o.predict(model.con, kag.h2o)$predict)

result.kag.con<-exp(result.kag.con*(a-b)+b)-1
write.csv(result.kag.con,"sub_continueonly2.csv")



###stack continue variable model:neural NN, catergory variable model:boosting
sub.train<-read.csv("./submodel/con_cat_train.csv")
sub.test<-read.csv("./submodel/con_cat_test.csv")

sub.train$loss<-log(data$loss+1)[train]

library(VIM)
model.lm<-lm(loss~.,data = sub.train)
summary(model.lm)
plot(model.lm)
result.lm<-predict(model.lm,sub.test)
write.csv(result.lm,"kagsub.lmstack.csv")



sub.train.nom<-as.data.frame(lapply(sub.train,normalize))
sub.test.nom<-as.data.frame(lapply(sub.test,normalize))


subtrain.h2o<-as.h2o(sub.train.nom)
subtest.h2o<-as.h2o(sub.test.nom)
model.nn<-h2o.deeplearning(x=1:2,
                            y=3,
                            training_frame = subtrain.h2o,
                            activation = "TanhWithDropout",
                            input_dropout_ratio = 0.2, # % of inputs dropout
                            hidden_dropout_ratios = c(0.5,0.5), # % for nodes dropout
                            #                        balance_classes = TRUE, 
                            hidden = c(100,50), # three layers of 50 nodes
                            epochs = 500,# max. no. of epochs
                            nfolds = 10)


a=max(log(sub.train$loss+1)) ###11.57931228
a
b=min(log(sub.train$loss+1))  ###0.5128236264

yhat_train <- as.data.frame(h2o.predict(model.nn,subtrain.h2o)$predict)
yhat_train<-exp(yhat_train*(a-b)+b)+1

write.csv(yhat_train,"con_train.csv")

sum((yhat_train-as.data.frame(train.h2o$loss)$loss)^2)/nrow.H2OFrame(train.h2o)


result.kag.nn<-as.data.frame(h2o.predict(model.nn,subtest.h2o)$predict)

result.kag.nn<-exp(result.kag.nn*(a-b)+b)-1

write.csv(result.kag.nn,"kagsub.nnstack.csv")




plot(x=as.data.frame(predict)[,1],y=data$loss[-train])
