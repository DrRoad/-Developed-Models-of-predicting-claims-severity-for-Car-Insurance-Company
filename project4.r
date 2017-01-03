library(ggplot2)
library(car)
library(dplyr)
library(corrplot)
library(caret)
library(gbm)
set.seed(0)
data<-read.csv("train.csv")
train<-sample(1:nrow(data),0.8*nrow(data))
data.train<-data[train,-1]
data.test<-data[-train,-1]
loss.train<-log(data$loss[train])
loss.test<-log(data$loss[-train])
colname<-names(data)
length(colname)

###sebset catergory data
data.t.cat<- data.train[,sapply(data.train,function(x) ifelse(class(x)=="factor",TRUE,FALSE))]
data.t.con<- data.train[,sapply(data.train,function(x) ifelse(!class(x)=="factor",TRUE,FALSE))]
data.t.con$loss<-log(data.t.con$loss+1)
scatterplotMatrix(data.t.con[1:1000,])
model.lm <- lm(loss~ .,data = data.t.con)
summary(model.lm)
plot(model.lm)
vif(model.lm)
model.lm.re<-lm(loss~. -cont12 -cont6 -cont1,data = data.t.con)
vif(model.lm.re)
summary(model.lm.re)
cor(data.t.con)  ###11,12 0.99 correlation;cont 1-9 0.931;1-10, 0.81; cont6-1 0.76; cont6-7 0.66; con6-9 0.8;con6-10,0.88;
###con9-10,0.787, con11-6,11-7,11-9;con12-1,12-6 0.8
corrs<-cor(data.t.con,method = "pearson")
corrplot.mixed(corrs,upper = "square",order="hclust")

###create dummy variables and preprocess data


dm.train.pre<-model.matrix(loss~ .,data=data.train)
dm.test.pre<-model.matrix(loss~ .,data=data.test)

pre<-preProcess(dm.train.pre, method = "nzv")
pre

dm.train<-predict(prep,dm.train)
dm.test<-predict(prep,dm.test)

##write.csv(dm.train,"dm.train.csv")
##write.csv(dm.test,"dm.test.csv")

dm.train<-read.csv("dm.train.csv")[,-1]
dm.test<-read.csv("dm.test.csv")[,-1]

dm.train.cat<-select(dm.train,contains("cat"))
wss = function(data, nc = 15, seed = 0) {
  wss = (nrow(data) - 1) * sum(apply(data, 2, var))
  for (i in 2:nc) {
    set.seed(seed)
    wss[i] = sum(kmeans(data, centers = i, iter.max = 100, nstart = 100)$withinss)
  }
  return(wss)
}
Ktune<-wss(dm.train.cat,20)
plot(1:20, Ktune, type = "b",
     xlab = "Number of Clusters",
     ylab = "Within-Cluster Variance",
     main = "Scree Plot for the K-Means Procedure")
names(dm.train)
model.lm.dm<-lm(Loss~ .-cont12 -cont6 -cont1, data =dm.train)
summary(model.lm.dm)



###### PCA
library(psych)
dm.tr
model.pca<-fa.parallel(dm.train[,-c(1,141,146,152)],fa="pc",n.iter = 10)
model.pca.50<- principal(dm.train[,-c(1,141,146,149,150,152)],nfactors = 48,rotate = "none")


####Gradient boosting machine
fitCtrl<-trainControl(method = "cv",
                      number = 10,
                      verboseIter = TRUE,
                      summaryFunction = defaultSummary)
gbmGrid<-expand.grid(n.trees=seq(100,10000,100),
                     interaction.depth=5,
                     shrinkage=0.1,
                     n.minobsinnode=50)
gbmFit<-train(x=dm.train,
              y=loss.train,
              method="gbm",
              trControl=fitCtrl,
              tuneGrid=gbmGrid,
              metric="RMSE",
              maximize = FALSE)
predict.test<-predict(gbmFit,dm.test)
predict.test
RMSE(pred = predict.test, obs = loss.test)
sum((predict.test-loss.test)^2)/(2*length(loss.test))


kag.test<-read.csv("test.csv")[,-1]
dm.kag.test<-model.matrix(~.,kag.test)
kag.test.dm<-predict(preProc,dm.kag.test)
predict.kag.test<-predict(gbmFit,kag.test.dm)h


###2nd try
input<-cbind(as.data.frame(dm.train),loss.train)
model.gbm<-gbm(loss.train ~ ., data = input,
               distribution = "gaussian",
               n.trees =2600,
               interaction.depth = 5,
               shrinkage = 0.1)
save(model.gbm,file = "./model.gbm.rda")
kag.test<-read.csv("test.csv")[,-1]
dm.kag.test<-model.matrix(~.,kag.test)
kag.test.dm<-predict(prep,dm.kag.test)
predict.kag.test<-predict(model.gbm,newdata = as.data.frame(kag.test.dm), n.trees = 2600)
final<-exp(predict.kag.test)
final
write.csv(final,"submit.csv")

###3rd try ##2640 trees 
input.all.x<-rbind(as.data.frame(dm.train),as.data.frame(dm.test))
input.all.y<-c(loss.train,loss.test)
input<-cbind(input.all.x,input.all.y)
names(input$input.all.y)<-c("loss")
model.gbm.2640<-gbm(input.all.y ~ ., data = input,
                    distribution = "gaussian",
                    n.trees =2640,
                    interaction.depth = 5,
                    shrinkage = 0.1)
save(model.gbm.2640,file = "./model.gbm.2640.rda")
predict.kag.test.2640<-predict(model.gbm.2640,newdata = as.data.frame(kag.test.dm), n.trees = 2640)
final.2640<-exp(predict.kag.test.2640)
final.2640
write.csv(final.2640,"submit2640.csv")

###3rd try ##2640 trees, n.minobsinnode=10
input.all.x<-rbind(as.data.frame(dm.train),as.data.frame(dm.test))
input.all.y<-c(loss.train,loss.test)
input<-cbind(input.all.x,input.all.y)
names(input$input.all.y)<-c("loss")
model.gbm.n.min20<-gbm(input.all.y ~ ., data = input,
                    distribution = "gaussian",
                    n.trees =2640,
                    n.minobsinnode = 20,
                    interaction.depth = 5,
                    shrinkage = 0.1)
save(model.gbm.n.min20,file = "./model.gbm.n.20.rda")
predict.kag.test.n.min20<-predict(model.gbm.n.min20,newdata = as.data.frame(kag.test.dm), n.trees = 2640)
final.n.min20<-exp(predict.kag.test.n.min20)
final.2640
write.csv(final.n.min20,"submitnmin20.csv")

####4th try
model.gbm.n.min15<-gbm(input.all.y ~ ., data = input,
                       distribution = "gaussian",
                       n.trees =2640,
                       n.minobsinnode = 20,
                       interaction.depth = 5,
                       shrinkage = 0.1)
save(model.gbm.n.min20,file = "./model.gbm.n.20.rda")
predict.kag.test.n.min20<-predict(model.gbm.n.min20,newdata = as.data.frame(kag.test.dm), n.trees = 2640)
final.n.min20<-exp(predict.kag.test.n.min20)
final.2640
write.csv(final.n.min20,"submitnmin20.csv")

###try log(y+1)
input$input.all.y<-data$loss
ggplot()+geom_histogram(aes(x=5*(log(input$input.all.y)+200)))
model.gbm.yt<-gbm(input.all.y ~ ., data = input,
                       distribution = "gaussian",
                       n.trees =2640,
                       n.minobsinnode = 20,
                       interaction.depth = 5,
                       shrinkage = 0.1)
save(model.gbm.n20,file = "./model.gbm.n20.no.y.trans.rda")
predict.kag.test<-predict(model.gbm.n20,newdata = as.data.frame(kag.test.dm), n.trees = 2640)
predict.kag.test
write.csv(predict.kag.test,"submitnnoytrans.csv")


###try (log(y)+200)
input$input.all.y<-log(log(data$loss+200))
x<-log(data$loss)
ggplot()+geom_histogram(aes(x=x,fill= "red"),bins=100)+labs(title="Y Transformation",y="Population",x="Log(loss)")+xlim(c(4,12))
model.gbm.yt<-gbm(input.all.y ~ ., data = input,
                  distribution = "gaussian",
                  n.trees =2640,
                  n.minobsinnode = 20,
                  interaction.depth = 5,
                  shrinkage = 0.1)
save(model.gbm.yt,file = "./model.gbm.yt.rda")
predict.kag.test<-predict(model.gbm.yt,newdata = as.data.frame(kag.test.dm), n.trees = 2640)
predict.kag.test
final<-exp(predict.kag.test.2640)-200
write.csv(final,"submitytrans.csv")
###



###redummy
data.full<-rbind(data[,-c(1,132)],kag.test)
dm.all<-model.matrix(~.,data.full)
dm.all.train<-dm.all[1:188318,]
dm.all.train<-mutate(dm.all.train,loss=exp(data$loss+200))
dm.all.test<-as.data.frame(dm.all[188319:313864,])

preProc<-preProcess(dm.all.train, method = "nzv",freqCut = 99/1)

model.gbm.all<-gbm(loss ~ ., data = dm.all.train,
                   distribution = "gaussian",
                   n.trees =2640,
                   n.minobsinnode = 20,
                   interaction.depth = 5,
                   shrinkage = 0.1)
###library(dplyr)
####library(corrplot)
##corrs <- cor(as_train %>% select(contains("cont")), method = "pearson")
##corrplot.mixed(corrs, upper = "square", order="hclust"
##dm_train <- model.matrix(loss ~ ., data = train_e)
##preProc <- preProcess(dm_train,method = "nzv")
##preProc
##dm_train <- predict(preProc, dm_train)
##dim(dm_train)
##set.seed(321)
##trainIdx <- createDataPartition(loss_e, 
##                               p = .8,
##                               list = FALSE,
##                               times = 1)
##subTrain <- dm_train[trainIdx,]
##subTest <- dm_train[-trainIdx,]
#lossTrain <- loss_e[trainIdx]
#lossTest <- loss_e[-trainIdx]
##lmFit <- train(x = subTrain, y = lossTrain,method = "lm")
##lmImp <- varImp(lmFit, scale = FALSE)
##plot(lmImp,top = 20)
mean(lmFit$resample$RMSE)
predicted <- predict(lmFit, subTest)
RMSE(pred = predicted, obs = lossTest)
plot(x = predicted, y = lossTest)
##The function trainControl can be used to specifiy the type of resampling:
fitCtrl <- trainControl(method = "cv",
                        number = 5,
                        verboseIter = TRUE,
                        summaryFunction=defaultSummary)
###The tuning parameter grid can be specified by the user and pass to the train via argument tuneGrid
gbmGrid <- expand.grid( n.trees = seq(100,500,50), 
                        interaction.depth = c(1,3,5,7), 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)
gbmFit <- train(x = subTrain, 
                y = lossTrain,
                method = "gbm", 
                trControl = fitCtrl,
                tuneGrid = gbmGrid,
                metric = 'RMSE',
                maximize = FALSE)
##PLOTTING THE RESAMPLING PROFILE
plot(gbmFit)
plot(gbmFit, plotType = "level")
gbmImp <- varImp(gbmFit, scale = FALSE)
plot(gbmImp,top = 20)
mean(gbmFit$resample$RMSE)

plot(gbmFit)
ggplot()+geom_line(aes(x=c(5,10,20,50),y=c(1165.24778,1162.56392,1162.22589,2251.57822)))+labs(title="Tune of n.minobsinnode",x="n.minobsinnode",
                                                                                               y="Kaggle Score")
