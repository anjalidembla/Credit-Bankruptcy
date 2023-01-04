library(aod)
library(caret)
library(caTools)
library(dplyr)
library(ggplot2)
library(ISLR)
library(mlbench)

library(plyr)
library(readr)

library(stats)
library(MASS)
library(readxl)

###Explore data set###
credit<- read_excel(file.choose())


#Initial Observations
#12 car0 and 337 car - Changed car0 to car
#no other NA values found

dim(credit)
summary(credit)

###Split data###
set.seed(1000)
credit$default <- as.factor(credit$default)

Training <- createDataPartition(credit$default, p=0.8, list=FALSE)
training <- credit[ Training, ]
testing <- credit[ -Training, ]

preproc <- c("center", "scale")
control <- trainControl(method = "cv", number = 10, savePredictions = TRUE)   

########################## LOGISTIC MODEL ##########################

mod_log_t1 <- train(default ~ checking_balance + months_loan_duration + credit_history + purpose
                    + amount + savings_balance + employment_duration + percent_of_income
                    + years_at_residence + age + other_credit + housing + existing_loans_count
                    + job + dependents + phone,
                    data=training, method="glm", family="binomial",
                    metric = "Accuracy",
                    trControl = control, preProcess = preproc)

summary(mod_log_t1)
pred_log_t1 = predict(mod_log_t1, newdata=testing)

confusionMatrix(data=pred_log_t1, testing$default, mode = 'prec_recall')
mod_log_t2 <- train(default ~ checking_balance + months_loan_duration + credit_history + purpose
                    + amount + savings_balance + employment_duration + percent_of_income
                    + years_at_residence + age ,
                    data=training, method="glm", family="binomial",
                    metric = "Accuracy",
                    trControl = control, preProcess = preproc)

summary(mod_log_t2)
pred_log_t2 = predict(mod_log_t2, newdata=testing)

confusionMatrix(data=pred_log_t2, testing$default, mode = 'prec_recall')

mod_log_t3 <- train(default ~ checking_balance + months_loan_duration + credit_history 
                    + amount + savings_balance + employment_duration ,
                    data=training, method="glm", family="binomial",
                    metric = "Accuracy",
                    trControl = control, preProcess = preproc)

summary(mod_log_t3)
pred_log_t3 = predict(mod_log_t3, newdata=testing)

confusionMatrix(data=pred_log_t3, testing$default, mode = 'prec_recall')

#Taking model 2 as it has the lowest AIC
########################## LDA MODEL ##########################

mod_lda <- train(default ~ checking_balance + months_loan_duration + credit_history + purpose
                    + amount + savings_balance + employment_duration + percent_of_income
                    + years_at_residence + age + other_credit + housing + existing_loans_count
                    + job + dependents + phone,  
                    data = training, method = "lda", family="binomial",                 
                    metric = "Accuracy",   
                    trControl = control, preProcess = preproc)                 

mod_lda                                               
pred_lda = predict(mod_lda, newdata=testing)

confusionMatrix(data=pred_lda, testing$default, mode = 'prec_recall')                 

########################## QDA MODEL ##########################

mod_qda <- train(default ~ checking_balance + months_loan_duration + credit_history + purpose
                    + amount + savings_balance + employment_duration + percent_of_income
                    + years_at_residence + age + other_credit + housing + existing_loans_count
                    + job + dependents + phone,     
                    data = training, method = "qda", family="binomial",  
                    metric = "Accuracy",                          
                    trControl = control, preProcess = preproc)      

mod_qda
pred_qda = predict(mod_qda, newdata=testing)

confusionMatrix(data=pred_qda, testing$default, mode = 'prec_recall')                      

########################## KNN ##########################

mod_knn <- train(default ~ checking_balance + months_loan_duration + credit_history + purpose
                    + amount + savings_balance + employment_duration + percent_of_income
                    + years_at_residence + age + other_credit + housing + existing_loans_count
                    + job + dependents + phone,     
                    data = training, method = "knn",  
                    metric = "Accuracy",                          
                    trControl = control, preProcess = preproc, tuneLength = 20)      
summary(mod_knn)
pred_knn = predict(mod_knn, newdata=testing)

confusionMatrix(data=pred_knn, testing$default, mode = 'prec_recall')    

########################## TREE ##########################

mod_tree = train(default ~ checking_balance + months_loan_duration + credit_history + purpose
                    + amount + savings_balance + employment_duration + percent_of_income
                    + years_at_residence + age + other_credit + housing + existing_loans_count
                    + job + dependents + phone,            
                    data=training,                 
                    method="rpart",                 
                    parms = list(split="gini"),     
                    metric = "Accuracy",       
                    trControl = control,    
                    tuneLength = 20)  

mod_tree
pred_tree = predict(mod_tree, newdata=testing)

confusionMatrix(data=pred_tree, testing$default, mode = 'prec_recall') 






