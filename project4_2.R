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
data.all.qua<-select(data.full,contains("con"))

###dummy catergory class
dm.all.cat<-model.matrix(~.,data.all.cat)
dm.train.cat<-dm.all.cat[1:188318,]
dm.test.cat<-dm.all.cat[188319:313864,]

### normalize quantitive 
normalize = function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}
data.all.qua.nom<-as.data.frame(lapply(data.all.qua,normalize))

loss.all.nom<-normalize(log(data$loss+1))
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
                     hidden = 1,
                     data = train.qua[train,])


#### catergory class

#dm.all.train<-mutate(dm.all.train,loss=log(data$loss))
dm.all.test<-as.data.frame(dm.all[188319:313864,])

preProc<-preProcess(dm.all.train, method = "zv")
preProc

save(preProc,file = "./preproczv.rda")
dm.all.pp<-predict(preProc,dm.all)
dm.all.train.pp<-as.data.frame(predict(preProc,dm.all.train))
dm.all.train.pp$loss<-log(data$loss)
