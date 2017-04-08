# import relevant packages and headers
rm(list = ls())

source("/Users/sam/All-Program/App/IIT-Projects/Loan-Defaulters/Data_Cleaner.R")
source("/Users/sam/All-Program/App/IIT-Projects/Loan-Defaulters/Feature_Selection.R")
library(caret)
library(dummy)
library(arm)
library(OOmisc)



#################   Data File Location    ##################
# Load the Data
datam = read.csv2("/Users/sam/All-Program/App-DataSet/IIT-Projects/Loan-Defaulters/LoanStats3a.csv", header=TRUE, sep=",", skip=1)


# Get The sample train and test dataset
smp_size <- floor(0.75 * nrow(datam))
print (smp_size)
set.seed(124)      # To repeat the sample random sample using seed
train_ind <- sample(seq_len(nrow(datam)), size = smp_size)
train <- datam[train_ind,]
test <- datam[-train_ind,]
write.csv(train, file = "/Users/sam/All-Program/App-DataSet/IIT-Projects/Loan-Defaulters/Training_data.csv")
write.csv(test, file = "/Users/sam/All-Program/App-DataSet/IIT-Projects/Loan-Defaulters/Test_data.csv")



#####################################################################################################
###################################### Prepare Data set #############################################
#####################################################################################################
# Load Data:
training_data = read.csv2("/Users/sam/All-Program/App-DataSet/IIT-Projects/Loan-Defaulters/Training_data.csv", header=TRUE, sep=",")

test_data = read.csv2("/Users/sam/All-Program/App-DataSet/IIT-Projects/Loan-Defaulters/Test_data.csv", header=TRUE, sep=",")

varm = read.csv2("/Users/sam/All-Program/App-DataSet/IIT-Projects/Loan-Defaulters/Variables-Sheet1.csv", header=TRUE, sep=",")

# Remove all the rows with response as "current"
dat_train <- subset(training_data, loan_status == 'Charged Off' | loan_status == 'Fully Paid')
dat_test <- subset(test_data, loan_status == 'Charged Off' | loan_status == 'Fully Paid')

# Data on Status:
feature_set1<-subset(varm, Status=="1")
feature_set1$LoanStatNew
feature_set0<-subset(varm, Status=="0")
feature_set0$LoanStatNew
feature_set2<-subset(varm, Status=="2")
feature_set2$LoanStatNew

# Get the data for the given feature set.
intrsct<-intersect(feature_set1$LoanStatNew, colnames(dat_train))
setdiff(feature_set1$LoanStatNew,intrsct) # Column names that are present in varm but not a part of datam column_names

# Finding the subset of datam with features decided as initial analysis
dat_train_1<-subset(dat_train, select = intrsct)
dat_test_1<-subset(dat_test, select = intrsct)
#head(dat_1)

# Clean columns 15-30 
# Remove Columns:
col_names <- c("id","application_type","annual_inc","home_ownership","inq_last_6mths","mths_since_last_delinq","out_prncp","pub_rec","pymnt_plan", "recoveries","revol_util","tax_liens","total_acc","total_pymnt_inv", "zip_code")

# Clean Training and Test Dataset
x_train_clean <- clean(dat_train_1)
x_train_clean <- drop_columns(x_train_clean,col_names)
x_train_clean <- var_cleaner(x_train_clean)         # Remove the irrelevant variables
x_train_clean <- na.omit(x_train_clean)

x_test_clean <- clean(dat_test_1)
x_test_clean <- drop_columns(x_test_clean,col_names)
x_test_clean <- var_cleaner(x_test_clean)         # Remove the irrelevant variables
x_test_clean <- na.omit(x_test_clean)

dim(x_train_clean)
dim(x_test_clean)


# # Get The sample train and test dataset
# smp_size <- floor(0.75 * nrow(x_clean))
# print (smp_size)
# set.seed(124)      # To repeat the sample random sample using seed
# train_ind <- sample(seq_len(nrow(x_clean)), size = smp_size)
# train <- x_clean[train_ind,]
# test <- x_clean[-train_ind,]
# #sample_x_clean <- x_clean[sample(1:nrow(x_clean), 20000, replace=FALSE), ]
# #write.csv(train, file = "/Users/sam/All-Program/App-DataSet/IIT-Projects/Insurance-Defaulters/Training_data.csv")
# #write.csv(test, file = "/Users/sam/All-Program/App-DataSet/IIT-Projects/Insurance-Defaulters/Test_data.csv")





#####################################################################################################
################################### Prepare the Dataset  ############################################
#####################################################################################################
# Prepare the final Training Data and testing data
train_y <- x_train_clean$loan_status.new
train_x <- x_train_clean[ , ! colnames(x_train_clean) %in% c("loan_status", "loan_status.new")]

test_y <- x_test_clean$loan_status.new
test_x <- x_test_clean[ , ! colnames(x_test_clean) %in% c("loan_status", "loan_status.new")]

standarize = 'No'
if (standarize == 'Yes'){
  data_tot <- rbind(train_x, test_x)
  train_x <- avg_standarize(data_tot,train_x)
  test_x <- avg_standarize(data_tot,test_x)
}


#write.csv(train, file = "/Users/sam/All-Program/App-DataSet/IIT-Projects/Insurance-Defaulters/Training_data.csv")
#write.csv(test, file = "/Users/sam/All-Program/App-DataSet/IIT-Projects/Insurance-Defaulters/Test_data.csv")






#####################################################################################################
###################### Normal features selection Logistic model  ####################################
#####################################################################################################

glm.out <- glm(train_y~acc_now_delinq+chargeoff_within_12_mths+delinq_2yrs+delinq_amnt+dti+earliest_cr_line+emp_length+grade+installment+int_rate+last_pymnt_amnt+loan_amnt+open_acc+out_prncp_inv+pub_rec_bankruptcies+revol_bal+term+total_pymnt+total_rec_late_fee+verification_status+out_prncp, family=binomial(logit), data=train_x)

#### AIC  ####
AIC_F1 <- step(glm.out, direction=c('both'),steps=2000)
summary(AIC_F1)

pred.AIC_F1 <- predict(AIC_F1, test_x, type="response") 

pred_AIC_F1 <- prediction_vector(pred.AIC_F1)  
mat_AIC_F1 <- table(pred_AIC_F1, test_y)
confusionMatrix(mat_AIC_F1)
 
#### BIC  ####
BIC_F1 <- step(glm.out, direction=c('both'), steps=2000, k=log(nrow(train_x)))
summary(BIC_F1)

pred.BIC_F1 <- predict(BIC_F1, test_x, type="response") 

pred_BIC_F1 <- prediction_vector(pred.BIC_F1)  
mat_BIC_F1 <- table(pred_BIC_F1, test_y)
confusionMatrix(mat_BIC_F1)





##################################################################################################### ###################### Interaction features selection Logistic model  ###############################
#####################################################################################################
inter_train_x <- prep_interaction(train_x)
inter_test_x <- prep_interaction(test_x)

glm_interaction.out <- glm(train_y~acc_now_delinq+chargeoff_within_12_mths+delinq_2yrs+delinq_amnt+dti+earliest_cr_line+emp_length+grade+installment+int_rate+last_pymnt_amnt+loan_amnt+open_acc+out_prncp_inv+pub_rec_bankruptcies+revol_bal+term+total_pymnt+total_rec_late_fee+verification_status+out_prncp+intr1+intr2+intr3+intr4+intr5+intr6+intr7+intr8+intr9+intr10+intr11+intr12+intr13+intr14+intr15+intr16+intr17+intr18+intr19+intr20+intr21, family=binomial(logit), data=inter_train_x)

#### AIC  ####
AIC_F2 <- step(glm_interaction.out, direction=c('both'),steps=2000)
summary(AIC_F1)

pred.AIC_F2 <- predict(AIC_F2, inter_test_x, type="response") 

pred_AIC_F2 <- prediction_vector(pred.AIC_F2)  
mat_AIC_F2 <- table(pred_AIC_F2, test_y)
confusionMatrix(mat_AIC_F2)

#### BIC  ####
BIC_F2 <- step(glm_interaction.out, direction=c('both'), steps=2000, k=log(nrow(train_x)))
summary(BIC_F2)

pred.BIC_F2 <- predict(BIC_F2, inter_test_x, type="response") 

pred_BIC_F2 <- prediction_vector(pred.BIC_F2)  
mat_BIC_F2 <- table(pred_BIC_F2, test_y)
confusionMatrix(mat_BIC_F2)





#####################################################################################################
###################### Polynomial features selection Logistic model  ###############################
#####################################################################################################
poly_train_x <- prep_polynomial(train_x)
poly_test_x <- prep_polynomial(test_x)

glm_polynomial.out <- glm(train_y~acc_now_delinq+chargeoff_within_12_mths+delinq_2yrs+delinq_amnt+dti+earliest_cr_line+emp_length+grade+installment+int_rate+last_pymnt_amnt+loan_amnt+open_acc+out_prncp_inv+pub_rec_bankruptcies+revol_bal+term+total_pymnt+total_rec_late_fee+verification_status+out_prncp+p1+p2+p3+p4+p5+p6+p7, family=binomial(logit), data=poly_train_x)

#### AIC ####
AIC_F3 <- step(glm_polynomial.out, direction=c('both'), steps=2000)
summary(AIC_F3)

pred.AIC_F3 <- predict(AIC_F3, poly_test_x, type="response") 

pred_AIC_F3 <- prediction_vector(pred.AIC_F3)  
mat_AIC_F3 <- table(pred_AIC_F3, test_y)
confusionMatrix(mat_AIC_F3)

#### BIC ####
BIC_F3 <- step(glm_polynomial.out, direction=c('both'), steps=2000, k=log(nrow(train_x)))
summary(BIC_F3)

pred.BIC_F3 <- predict(BIC_F3, poly_test_x, type="response") 

pred_BIC_F3 <- prediction_vector(pred.BIC_F3)  
mat_BIC_F3 <- table(pred_BIC_F3, test_y)
confusionMatrix(mat_BIC_F3)





#####################################################################################################
############################# Both Polynomical and Interactions  ####################################
#####################################################################################################
poly_intr_train_x <- prep_interaction(prep_polynomial(train_x))
poly_intr_test_x <- prep_interaction(prep_polynomial(test_x))

glm_polynomial_interaction.out <- glm(train_y~acc_now_delinq+chargeoff_within_12_mths+delinq_2yrs+delinq_amnt+dti+earliest_cr_line+emp_length+grade+installment+int_rate+last_pymnt_amnt+loan_amnt+open_acc+out_prncp_inv+pub_rec_bankruptcies+revol_bal+term+total_pymnt+total_rec_late_fee+verification_status+out_prncp+p1+p2+p3+p4+p5+p6+p7+intr1+intr2+intr3+intr4+intr5+intr6+intr7+intr8+intr9+intr10+intr11+intr12+intr13+intr14+intr15+intr16+intr17+intr18+intr19+intr20+intr21, family=binomial(logit), data=poly_intr_train_x)

save(glm_polynomial_interaction.out, file='/Users/sam/All-Program/App-DataSet/IIT-Projects/Loan-Defaulters/glm_polynomial_interaction.out')

#### AIC ####
AIC_F4 <- step(glm_polynomial_interaction.out, direction=c('both'), steps=2000)
summary(AIC_F4)
save(AIC_F4, file='/Users/sam/All-Program/App-DataSet/IIT-Projects/Loan-Defaulters/AIC_F4')

pred.AIC_F4 <- predict(AIC_F4, poly_intr_test_x, type="response") 

pred_AIC_F4 <- prediction_vector(pred.AIC_F4)  
mat_AIC_F4 <- table(pred_AIC_F4, test_y)
confusionMatrix(mat_AIC_F4)

#### BIC ####
BIC_F4 <- step(glm_polynomial_interaction.out, direction=c('both'), steps=2000, k=log(nrow(train_x)))
summary(BIC_F4)
save(AIC_F4, file='/Users/sam/All-Program/App-DataSet/IIT-Projects/Loan-Defaulters/AIC_F4')

pred.BIC_F4 <- predict(BIC_F4, poly_intr_test_x, type="response") 

pred_BIC_F4 <- prediction_vector(pred.BIC_F4)  
mat_BIC_F4 <- table(pred_BIC_F4, test_y)
confusionMatrix(mat_BIC_F4)




#####################################################################################################
#################################### FINAL AIC AND BIC MODEL  #######################################
#####################################################################################################

### FINAL AIC MODEL ###
final.AIC<- glm(formula = train_y ~ dti + emp_length + grade + installment + 
                  int_rate + last_pymnt_amnt + loan_amnt + pub_rec_bankruptcies + 
                  revol_bal + term + total_pymnt + total_rec_late_fee + p2 + 
                  p3 + p4 + p5 + p6 + p7 + intr2 + intr4 + intr5 + intr7 + 
                  intr10 + intr11 + intr12 + intr13 + intr14 + intr15 + intr19 + 
                  intr20 + intr21, family = binomial(logit), data = poly_intr_train_x)
save(final.AIC, file='/Users/sam/All-Program/App-DataSet/IIT-Projects/Loan-Defaulters/final.AIC')

AIC_F4 <- step(final.AIC, direction=c('both'), steps=2000)
summary(AIC_F4)
save(AIC_F4, file='/Users/sam/All-Program/App-DataSet/IIT-Projects/Loan-Default/AIC_F4')

pred.AIC_F4 <- predict(AIC_F4, poly_intr_test_x, type="response") 

pred_AIC_F4 <- prediction_vector(pred.AIC_F4)  
mat_AIC_F4 <- table(pred_AIC_F4, test_y)
confusionMatrix(mat_AIC_F4)



### FINAL BIC MODEL ###
final.BIC<-glm(formula = train_y ~ installment + int_rate + last_pymnt_amnt + 
                 loan_amnt + term + total_pymnt + total_rec_late_fee + p2 + 
                 p4 + p7 + intr5 + intr10 + intr11 + intr12 + intr13 + intr14 + 
                 intr15 + intr19, family = binomial(logit), data = poly_intr_train_x)
save(final.BIC, file='/Users/sam/All-Program/App-DataSet/IIT-Projects/Loan-Defaulters/final.BIC')

BIC_F4 <- step(final.BIC, direction=c('both'), steps=2000, k=log(nrow(train_x)))
summary(BIC_F4)
save(BIC_F4, file='/Users/sam/All-Program/App-DataSet/IIT-Projects/Loan-Defaulters/BIC_F4')

pred.BIC_F4 <- predict(BIC_F4, poly_intr_test_x, type="response") 

pred_BIC_F4 <- prediction_vector(pred.BIC_F4)  
mat_BIC_F4 <- table(pred_BIC_F4, test_y)
confusionMatrix(mat_BIC_F4)





#####################################################################################################
##################################### MODEL DIAGNOSTICS  ############################################
#####################################################################################################
library(arm)
library(OOmisc)

# both_n_AIC <- AIC_F1
bothAIC<-final.AIC
bothBIC<-final.BIC

#### For the normal model (without polynomial and Interactions) ####
# pred.n_AIC <- predict(AIC_F1, train_x, type="response") 
resid.n_AIC <- resid(both_n_AIC)
binnedplot(pred.n_AIC,resid.n_AIC,nclass=300)

#### For the Final model (with polynomial and Interactions) ####
###Binned Residulas###
pred.AIC <- predict(bothAIC, poly_intr_train_x, type="response") 
resid.AIC <- resid(bothAIC)
binnedplot(pred.AIC,resid.AIC,nclass=300)

pred.BIC <- predict(bothBIC, poly_intr_train_x, type="response")
resid.BIC <- resid(bothBIC)
binnedplot(pred.BIC,resid.BIC,nclass=300) 

###Error Rate###
error.rateAIC<- mean((pred.AIC>0.5 & train_y==0) | (pred.AIC<0.5 & train_y==1))
error.rateAIC
summary(train_y)

###Expected Proportion Correctly Predicted (ePCP) ###
null.model<-glm.out <- glm(train_y~1, family=binomial(logit), data=train_x)
pred.null <- predict(null.model, train_x, type="response") 

ePCP(pred.null,train_y,alpha=0.05)
ePCP(pred.AIC,train_y,alpha=0.05)
ePCP(pred.BIC,train_y,alpha=0.05)

###Deviance###
bothAIC$null.deviance
bothAIC$deviance

bothBIC$null.deviance
bothBIC$deviance

###ROC Curve###
### The area measures discrimination, that is, the ability of the test ###
### to correctly classify "y" ###

install.packages('ROCR')
library(ROCR)

probAIC <- prediction(pred.AIC, train_y)
tprfprAIC<- performance(probAIC ,'tpr','fpr')
tprAIC <- unlist(slot(tprfprAIC, "y.values"))
fprAIC<- unlist(slot(tprfprAIC, "x.values"))
rocAIC <- data.frame(tprAIC, fprAIC)

probBIC <- prediction(pred.BIC, train_y)
tprfprBIC<- performance(probBIC ,'tpr','fpr')
tprBIC <- unlist(slot(tprfprBIC, "y.values"))
plot(tprfprBIC, colour="green")
fprBIC<- unlist(slot(tprfprBIC, "x.values"))
rocBIC <- data.frame(tprBIC, fprBIC)

abline(a=0, b=1, untf=FALSE)
plot(tprfprAIC, add=TRUE, colorize=TRUE)
plot(tprfprBIC, add=TRUE, colorize=TRUE)


