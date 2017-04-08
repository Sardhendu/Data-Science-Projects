# fit <- lm(y~annual_inc+delinq_2yrs+delinq_2yrs+dti+earliest_cr_line+emp_length+grade+home_ownership+inq_last_6mths+installment+int_rate+last_pymnt_amnt+loan_amnt+mths_since_last_delinq+open_acc+pub_rec+recoveries+revol_bal+revol_util+term+total_acc+total_pymnt+total_pymnt_inv+total_rec_int+total_rec_late_fee+total_rec_prncp+verification_status+zip_code, data=x)
#
#
# fit <- lm(x$int_rate~annual_inc, data=x)
#
# fit <- lm(x$int_rate~annual_inc+delinq_2yrs+delinq_2yrs+dti+earliest_cr_line+emp_length+grade+home_ownership+inq_last_6mths+installment+last_pymnt_amnt+loan_amnt+mths_since_last_delinq+open_acc+pub_rec+recoveries+revol_bal+revol_util+term+total_acc+total_pymnt+total_pymnt_inv+total_rec_int+total_rec_late_fee+total_rec_prncp+verification_status, data=x)



# ######### Check by fiiting a simple model  #############
#loan_status1 <- predict(dummyVars(~loan_status, data = sample_x_clean), newdata = sample_x_clean)
# glm.out <- glm(y~., family=binomial(logit), data=x)
# glm.out <- glm(y~acc_now_delinq+delinq_2yrs+dti+earliest_cr_line+emp_length+grade+home_ownership+inq_last_6mths+installment+int_rate+last_pymnt_amnt+loan_amnt+mths_since_last_delinq+open_acc+pub_rec+recoveries+revol_bal+revol_util+term+total_acc+total_pymnt+total_pymnt_inv+total_rec_int+total_rec_late_fee+total_rec_prncp+verification_status, family=binomial(logit), data=x)
#
# glm.out <- glm(y~acc_now_delinq+delinq_2yrs+dti+earliest_cr_line+emp_length+grade+home_ownership+inq_last_6mths+installment+int_rate+last_pymnt_amnt+loan_amnt+mths_since_last_delinq+open_acc+pub_rec+recoveries+revol_bal+revol_util+term+total_acc+total_pymnt+total_pymnt_inv+total_rec_int+total_rec_late_fee+total_rec_prncp+verification_status, family=binomial(logit), data=x)


# glm.out <- glm(y~acc_now_delinq+chargeoff_within_12_mths+delinq_2yrs+delinq_amnt+dti+earliest_cr_line+emp_length+grade+installment+int_rate+last_pymnt_amnt+loan_amnt+open_acc+out_prncp_inv+pub_rec_bankruptcies+revol_bal+term+total_pymnt+total_rec_late_fee+verification_status+out_prncp, family=binomial(logit), data=x)
#
#
# # total_rec_int, total_rec_prncp,total_rec_late_fee+total_rec_prncp+verification_status
# backwardAIC <- step(glm.out)
# summary(backwardAIC)
#
# glm.out <- glm(y~acc_now_delinq+chargeoff_within_12_mths+delinq_2yrs+delinq_amnt+dti+earliest_cr_line+emp_length+grade+installment+int_rate+last_pymnt_amnt+loan_amnt+open_acc+out_prncp_inv+pub_rec_bankruptcies+revol_bal+term+total_pymnt+total_rec_late_fee+verification_status+out_prncp, family=binomial(logit), data=x)
#
#
# cor(y,x$home_ownership)
# tbl <- table(y, x$home_ownership)
# chisq.test(tbl)


######### Analysis  ##########

# Drop : anual_inc
# Keep : delinq_2yrs, dti, grade(subgrade), int_rate
# Maybe: earliest_cr_line, home_ownership
# Doesn't make sense: emp_length



##########  ##########  ##########  ##########  ##########  ##########  ##########  ##########
##########  ##########  ##########  ##########  ##########  ##########  ##########  ##########
##########  ##########  ##########  ##########  ##########  ##########  ##########  ####################  ##########  ##########  ##########  ##########  ##########  ##########  ##########





# glm.out = glm(sample_x_clean$loan_status.new~sample_x_clean$dti, family=binomial(logit), data=sample_x_clean)
#
# summary(glm.out)




#fit <- lm(y~acc_now_delinq+annual_inc+chargeoff_within_12_mths+delinq_2yrs+delinq_2yrs+delinq_amnt+dti+earliest_cr_line+emp_length+grade+home_ownership+inq_last_6mths+installment+int_rate+last_pymnt_amnt+loan_amnt+mths_since_last_delinq+open_acc+out_prncp+out_prncp_inv+pub_rec+pub_rec_bankruptcies+pymnt_plan+recoveries+revol_bal+revol_util+tax_liens+term+total_acc+total_pymnt+total_pymnt_inv+total_rec_int+total_rec_late_fee+total_rec_prncp+verification_status+zip_code, data=sample_x_clean)


#glm.out <- glm(y~annual_inc+delinq_2yrs+delinq_2yrs+dti+earliest_cr_line+emp_length+grade+home_ownership+inq_last_6mths+installment+int_rate+last_pymnt_amnt+loan_amnt+mths_since_last_delinq+open_acc+pub_rec+recoveries+revol_bal+revol_util+term+total_acc+total_pymnt+total_pymnt_inv+total_rec_int+total_rec_late_fee+total_rec_prncp+verification_status, family=binomial(logit), data=x)



# lm(formula = lnWeight ~ lnLength, data = alligator)
#
# "acc_now_delinq"           "annual_inc"               "application_type"
# [4] "chargeoff_within_12_mths" "delinq_2yrs"              "delinq_amnt"
# [7] "dti"                      "earliest_cr_line"         "emp_length"
# [10] "grade"                    "home_ownership"           "inq_last_6mths"
# [13] "installment"              "int_rate"                 "last_pymnt_amnt"
# [16] "loan_amnt"                "loan_status"              "mths_since_last_delinq"
# [19] "open_acc"                 "out_prncp"                "out_prncp_inv"
# [22] "pub_rec"                  "pub_rec_bankruptcies"     "pymnt_plan"
# [25] "recoveries"               "revol_bal"                "revol_util"
# [28] "tax_liens"                "term"                     "total_acc"
# [31] "total_pymnt"              "total_pymnt_inv"          "total_rec_int"
# [34] "total_rec_late_fee"       "total_rec_prncp"          "verification_status"
# [37] "zip_code"                 "loan_status.new"


######### Create Dummy Variables  ###########

# grade <- predict(dummyVars(~grade, data = sample_x_clean), newdata = sample_x_clean)
# application_type <- predict(dummyVars(~application_type, data = sample_x_clean), newdata = sample_x_clean)
# home_ownership <- predict(dummyVars(~home_ownership, data = sample_x_clean), newdata = sample_x_clean)
# pymnt_plan <- predict(dummyVars(~pymnt_plan, data = sample_x_clean), newdata = sample_x_clean)
# grade <- predict(dummyVars(~grade, data = sample_x_clean), newdata = sample_x_clean)




# levels(x_clean$grade)
# levels(x_clean$application_type)
# levels(x_clean$home_ownership)
# levels(x_clean$mths_since_last_delinq)      # To check
# levels(x_clean$out_prncp)                   # Not Categorical
# levels(x_clean$out_prncp_inv)               # Not Categorical
# levels(x_clean$pub_rec)                     # To Check
# levels(x_clean$pub_rec_bankruptcies)        # To Check
# levels(x_clean$pymnt_plan)
# levels(x_clean$tax_liens)                   # To Check
# levels(x_clean$term)
# levels(x_clean$total_rec_late_fee)          # Not Categorical
# levels(x_clean$verification_status)
# levels(x_clean$zip_code)
# levels(x_clean$loan_status)






######### Rough #########
# library(ggplot2)
# mysvm      <- svm(Species ~ ., iris)
# Predicted  <- predict(mysvm, iris)
#
# mydf = cbind(iris, Predicted)
# qplot(sample_x_clean$annual_inc, sample_x_clean$earliest_cr_line, colour = Species, shape = Predicted,
#       data = iris)
#
# attach(sample_x_clean); plot(sample_x_clean$annual_inc, sample_x_clean$earliest_cr_line, col=c("red","blue")[sample_x_clean$loan_status]); detach(sample_x_clean)
#
#
#
# n = c(1, 2, 3,4,5)
# s = c("a", "a", "b", "c", "c")
# b = c(TRUE, FALSE, TRUE)
# df = data.frame(n, s)
# predict(dummyVars(~s, data = df), newdata = df)
#
# aaa <- dummy(iris$Species)




#x_col <- as.numeric(as.character(x[col_name]))
#x_col[is.na(x[col_name])] <- median.default(x_col,na.rm=TRUE)
#x_col_name
# print (median(x[col_name]))
#summary1<- (summary(x)) #$col_name))
#col_summary <- summary[]
#print (summary_col)


########### Analysis  ##########

# Look data for the given columns
#sel_col<-datam[c("loan_amnt","funded_amnt","installment", "annual_inc")]
# Top 5 elements:
#head(sel_col)

#colnames(datam)
#head(datam[15:30])
#unique(datam$verification_status)
#unique(datam$pymnt_plan)
#URL and description not needed.
#unique(datam$purpose)
#unique(datam$title)