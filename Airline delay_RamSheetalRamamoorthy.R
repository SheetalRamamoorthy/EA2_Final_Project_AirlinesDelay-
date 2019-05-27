library(ggplot2)
library(reshape2)
library(caret)
library(Matrix)
library(rpart)
library(rpart.plot)
library(randomForest)
library(xgboost)
library(data.table)
library(mlr)

############# DATA PREPARATION ###############

airline=read.csv(file = "C:\\Users\\SheetalRamamoorthy\\Documents\\EA2\\FINAL PROJECT\\Project_Pitch_2nd Attempt\\Flight-delays\\flights.csv",header = T,sep = ",")
setsize=floor(0.025*nrow(airline))
set.seed(123)
rowIndices <- sample(seq_len(nrow(airline)), size = setsize)
Airlines <- airline[rowIndices, ]
Airlines[1:10,]
nrow(Airlines)
ncol(Airlines)
write.csv(Airlines,"Airlines1.csv")  # used this data for the tableau EXLORATORY DATA ANALYSIS 
summary(Airlines)
 
# Data discovery 
# number of flights cancelled and number of flights not cancelled  
table(Airlines$CANCELLED) # 1 is cancelled and 0 not cancelled 
# number of diverted flights
table(Airlines$DIVERTED) # 1 is diverted 
# number of flights delayed 
table(Airlines$ARRIVAL_DELAY<0)  

# Exploratory Data analysis 
# Dealing With the  Missing Data And  NA's   

completeFun <- function(data, desiredCols) {
  completeVec <- complete.cases(data[, desiredCols])  # function created to remove the NA's
  return(data[completeVec, ])
}

air=completeFun(Airlines, "DEPARTURE_TIME")   # The NA's in the dearture time and the arrival delay are
#                                               linked iwth the NA's in other columns
summary(air)

air=completeFun(Airlines, "ARRIVAL_DELAY")
summary(air)

# removing the column  "YEAR" and "Cancellation reson" and Cancelled
air$YEAR=NULL
air$CANCELLED=NULL
air$CANCELLATION_REASON=NULL

# changing the NA's to ZERO in the columns "AIR_SYSTEM_DELAY,SECURITY_DELAY,AIRLINE_DELAY,
# LATE_AIRCRAFT_DELAY,WEATHER_DELAY".  
air$AIR_SYSTEM_DELAY[is.na(air$AIR_SYSTEM_DELAY)] = 0
air$SECURITY_DELAY[is.na(air$SECURITY_DELAY)]=0
air$AIRLINE_DELAY[is.na(air$AIRLINE_DELAY)]=0
air$LATE_AIRCRAFT_DELAY[is.na(air$LATE_AIRCRAFT_DELAY)]=0
air$WEATHER_DELAY[is.na(air$WEATHER_DELAY)]=0
summary(air)

# CREATING A NEW VARIABLE  "DEALYEDARRIVAL"
air$DELAYEDARRIVAL[air$ARRIVAL_DELAY<=0]<- 0  #  IT ARRIVED EARLY or on time 
# air$DELAYEDARRIVAL[air$ARRIVAL_DELAY<0]<-"ONTIME"  # ARRIVED ON TIME 
air$DELAYEDARRIVAL[air$ARRIVAL_DELAY>0]<-1 # IT HAD A DELAYED ARRIVAL
summary(air)

write.csv(air,"air.csv")   

# FIND THE CORRELATION ---Highly correlated variables
aircordata= air[,c(9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26,27,28,29)]
summary(aircordata)
correlation_mat <- round(cor(aircordata),2)
correlation_mat


highlyCorrelated <- findCorrelation(correlation_mat, cutoff=0.7,verbose = TRUE)
print(highlyCorrelated)   

# Heat map representing the correlation matrix 

melted_cormat <- melt(correlation_mat)
head(melted_cormat)
ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) +  geom_tile()

# Get upper triangle matrix
get_upper_tri <- function(correlation_mat){}
upper_tri <- get_upper_tri(correlation_mat)
upper_tri

# Melt the correlation matrix

melted_cormat <- melt(upper_tri, na.rm = TRUE) # using the reshape2 library 


# Heatmap of the correlation of the variables 
# using the person coefficients

ggplot(data = melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
   midpoint = 0, limit = c(-1,1), space = "Lab", name="Pearson\nCorrelation") +theme_minimal()+ 
    theme(axis.text.x = element_text(angle = 45, vjust = 1,  size = 12, hjust = 1))+coord_fixed()  # using the ggplot2 


# Splitting the AIR dataset into Training and Testing Dataset.
size <- floor(0.70 * nrow(air))  # 70% of the dataset for training the model
set.seed(321) #set a seed for being able to replicate
rowInd<- sample(seq_len(nrow(air)), size = size)
trainAir <- air[rowInd, ]
testAir <- air[-rowInd, ]

# Building the Predicitve Models 

# PREDICTIVE MODEL 1 
# DECISION TREES 

prop.table(table(trainAir$DELAYEDARRIVAL)) # in both the dataset the delayed arrival time is almost same around 36%
prop.table(table(testAir$DELAYEDARRIVAL))

# DECISION TREE MODEL 1
dtree=rpart(DELAYEDARRIVAL~ MONTH +DAY + DAY_OF_WEEK+
            SCHEDULED_DEPARTURE+DEPARTURE_TIME+DEPARTURE_DELAY+TAXI_OUT+WHEELS_OFF+AIR_TIME+DISTANCE+
            WHEELS_ON+TAXI_IN+SCHEDULED_ARRIVAL,data=trainAir,method = "class")
rpart.plot(dtree,extra = 106,varlen=0)

predict_dtree <-predict(dtree, testAir, type = 'class')
table=table(testAir$DELAYEDARRIVAL,predict_dtree)
names(dtree)
dtree$variable.importance
accuracy=sum(diag(table))/sum(table)   #accuracy 83.916%
accuracy

# DECISION TREE MODEL 2 
# WITH THE ORIGIN AIRPORT & DESTINATION AIRPORT VARIABLES 
dtree1=rpart(DELAYEDARRIVAL~ MONTH +DAY + DAY_OF_WEEK+
              DESTINATION_AIRPORT+ORIGIN_AIRPORT+WHEELS_ON+TAXI_IN+
              SCHEDULED_DEPARTURE+DEPARTURE_DELAY+TAXI_OUT+DISTANCE+
              SCHEDULED_ARRIVAL+AIR_TIME+DISTANCE,data=trainAir,method = "class")
rpart.plot(dtree1,extra = 106,varlen=0)
predict_dtree1 <-predict(dtree1, testAir, type = 'class')
table1=table(testAir$DELAYEDARRIVAL,predict_dtree1)

names(dtree1)
dtree1$variable.importance
accuracy1=sum(diag(table1))/sum(table1)   #accuracy value is 84.056%
accuracy1

# DECISION TREE MODEL 3 WITH IMPROVED ACCURACY 
# WITH THE ImpPORTANT variables and changing the hyperparameters

control <- rpart.control(minsplit = 4,minbucket = round(5 / 3),maxdepth = 3,cp = 0)
dtree2=rpart(DELAYEDARRIVAL~ DEPARTURE_DELAY+TAXI_OUT+WHEELS_ON+WHEELS_OFF+DEPARTURE_TIME+TAXI_IN+
               SCHEDULED_DEPARTURE,data=trainAir,method = "class",control = control)
rpart.plot(dtree2,extra=106,varlen=0)
predict_dtree2 <-predict(dtree2, testAir, type = 'class')
table2=table(testAir$DELAYEDARRIVAL,predict_dtree2)
dtree2$variable.importance
accuracy2=sum(diag(table2))/sum(table2)  # accuracy value is 84.242% 


# @@@@@@@@@@@@@@@@@!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@@@@@@@

# PREDICITVE MODEL 2 
# Random Forest 

# the dependant variable as a factor variable
trainAir$DELAYEDARRIVAL=as.factor(trainAir$DELAYEDARRIVAL)
testAir$DELAYEDARRIVAL=as.factor(testAir$DELAYEDARRIVAL)

# since the ORIGIN AND DESTINATION AIRPORT HAVE MORE THAN 53 LEVELS AND CANNOT BE USED TO PERFORM THE RANDOM FOREST EXPLAIN THAT AS WELL 
length(levels(trainAir$ORIGIN_AIRPORT))
set.seed(123)

# RANDOM FOREST MODEL 1 
rfmodel=randomForest(formula=DELAYEDARRIVAL~ MONTH +DAY+ DAY_OF_WEEK+SCHEDULED_DEPARTURE+DEPARTURE_DELAY+TAXI_OUT+
                       DISTANCE+SCHEDULED_ARRIVAL+AIR_TIME,data=trainAir,
                        ntree=100,importance =TRUE,replace=TRUE,mtry=9,maxnodes=500)
names(rfmodel)
rfmodel$confusion  # misclassification error rate for 0 =0.05,1=0.302
prediction_rf<- predict(object = rfmodel, newdata = testAir)

table_rf=table(testAir$DELAYEDARRIVAL,prediction_rf)
accuracy_RF=sum(diag(table_rf))/sum(table_rf)  # the accuracy is  85.102%
accuracy_RF

# RANDOM FOREST MODEL 2 WITH IMPROVED ACCURACY 
rfmodel1=randomForest(formula=DELAYEDARRIVAL~ MONTH +DAY + DAY_OF_WEEK+
                        SCHEDULED_DEPARTURE+DEPARTURE_TIME+DEPARTURE_DELAY+TAXI_OUT+WHEELS_OFF+AIR_TIME+DISTANCE+
                        WHEELS_ON+TAXI_IN+SCHEDULED_ARRIVAL,data=trainAir,
                     ntree=100,importance =TRUE,replace=TRUE,mtry=9,maxnodes=500)
names(rfmodel1)
rfmodel1$confusion  # misclassification error rate for 0 =0.05,1=0.2733 has reduced the misclassification error 
prediction_rf1<- predict(object = rfmodel1, newdata = testAir)
table_rf1=table(testAir$DELAYEDARRIVAL,prediction_rf1)
accuracy_RF1=sum(diag(table_rf1))/sum(table_rf1)   # accuracy 85.947%
accuracy_RF1

#@@@@@@@@@@@@@@@@@@@@@@@!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@@@@@@@
# PREDICTIVE MODEL 3 

# XGBOOST ALGORITHM 
# MODEL1 
train=trainAir
validation=testAir
options(na.action = 'na.pass')

new_tr=sparse.model.matrix(DELAYEDARRIVAL~MONTH +DAY + DAY_OF_WEEK+
                             SCHEDULED_DEPARTURE+DEPARTURE_TIME+DEPARTURE_DELAY+TAXI_OUT+WHEELS_OFF+AIR_TIME+DISTANCE+
                             WHEELS_ON+TAXI_IN+SCHEDULED_ARRIVAL-1,data = train,with=F)
train_label <- train$DELAYEDARRIVAL
train_label <- as.numeric(train_label)-1
d_train <- xgb.DMatrix(data = new_tr, label=train_label)

# validation 
new_val = sparse.model.matrix(DELAYEDARRIVAL~MONTH +DAY + DAY_OF_WEEK+
                                SCHEDULED_DEPARTURE+DEPARTURE_TIME+DEPARTURE_DELAY+TAXI_OUT+WHEELS_OFF+AIR_TIME+DISTANCE+
                                WHEELS_ON+TAXI_IN+SCHEDULED_ARRIVAL-1,data = validation, with = F)
val_label <- validation$DELAYEDARRIVAL
val_label <- as.numeric(val_label)-1
dval <- xgb.DMatrix(data = new_val, label=val_label)


params <- list(booster = "gbtree",objective = "binary:logistic",eta=0.3,gamma=0,max_depth=6,
               min_child_weight=1,subsample=1,colsample_bytree=1)

bst <- xgboost(data = d_train, label = train_label, max_depth = 2, eta = 1, nthread = 2,
               nrounds = 20,nfold=5,objective = "binary:logistic")
names(bst)
bst$evaluation_log
xgbpred <- round(predict(bst,dval))

table_xgb=table(testAir$DELAYEDARRIVAL,xgbpred)
accuracy_xgb=sum(diag(table_xgb))/sum(table_xgb)  # accuracy 85.89177%
accuracy_xgb

#view variable importance plot 
mat <- xgb.importance (feature_names = colnames(new_val),model = bst)   # show the important varaibles
xgb.plot.importance (importance_matrix = mat[1:29])

# XGBOOST MODEL 2 
# with the important variables from the plot 

train=trainAir
validation=testAir
options(na.action = 'na.pass')

new_tr1=sparse.model.matrix(DELAYEDARRIVAL~DEPARTURE_DELAY+TAXI_OUT+TAXI_IN+DISTANCE+AIR_TIME+MONTH-1,data = train,with=F)
train_label_xgb<- train$DELAYEDARRIVAL
train_label_xgb <- as.numeric(train_label_xgb)-1
d_train.xgb <- xgb.DMatrix(data = new_tr1, label=train_label_xgb)

# validation 
new_val.xgb = sparse.model.matrix(DELAYEDARRIVAL~DEPARTURE_DELAY+TAXI_OUT+TAXI_IN+DISTANCE+AIR_TIME+MONTH-1,data = validation, with = F)
val_label_xgb <- validation$DELAYEDARRIVAL
val_label_xgb <- as.numeric(val_label_xgb)-1
dval.xgb <- xgb.DMatrix(data = new_val.xgb, label=val_label_xgb)

#new_test=sparse.model.matrix(DELAYEDARRIVAL~.-1,data = testAir,replace= F)
#d_test=xgb.DMatrix(data = new_test)

params <- list(booster = "gbtree",objective = "binary:logistic",eta=0.3,gamma=0,max_depth=6,min_child_weight=1,subsample=1,colsample_bytree=1)

bst1 <- xgboost(data = d_train.xgb, label = train_label_xgb, max_depth = 2, eta = 1, nthread = 2, nrounds = 20,nfold=5,objective = "binary:logistic")

xgbpred1 <- round(predict(bst1,dval.xgb))
xgbpred1

table_xgb1=table(testAir$DELAYEDARRIVAL,xgbpred1)
accuracy_xgb1=sum(diag(table_xgb1))/sum(table_xgb1)  # accuracy 85.89177%

# @@@@@@@@@@@@@@@!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@@@@@@@@@@@@@
# PREDICTIVE MODEL 4 
#@@@@@@@@@
# LOGISTIC REGRESSION MODEL 1
 
log_model <- glm(DELAYEDARRIVAL ~MONTH +DAY + DAY_OF_WEEK+
                   SCHEDULED_DEPARTURE+DEPARTURE_TIME+DEPARTURE_DELAY+TAXI_OUT+WHEELS_OFF+AIR_TIME+DISTANCE+
                   WHEELS_ON+TAXI_IN+SCHEDULED_ARRIVAL,family=binomial,data=trainAir)
summary(log_model)
fit = predict(log_model,newdata=testAir,type='response')
fit = ifelse(fit > 0.5,1,0)

# rocr curve 
library(ROCR)
p <- predict(log_model, newdata=testAir, type="response")
pr <- prediction(p, testAir$DELAYEDARRIVAL)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc   # VALUE IS .9310603  TE MORE CLOSER THE VLAUE IS THE BETTER THE MODEL IS

# ACCURACY
table_logreg=table(testAir$DELAYEDARRIVAL,fit)
accuracy_logreg=sum(diag(table_logreg))/sum(table_logreg)  # accuracy is 86.3798%
accuracy_logreg

