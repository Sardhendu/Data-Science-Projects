


significant_features = c("dti","emp_length","gradeB","gradeC","gradeD","gradeE","gradeF","gradeG","installment","int_rate","last_pymnt_amnt","loan_amnt","open_acc","pub_rec_bankruptcies","term","total_pymnt","total_rec_late_fee")


###############   Using Interaction models   #################
prep_interaction <- function(data_in){
  data_in$intr1 <- I(data_in$dti*data_in$emp_length)
  data_in$intr2 <- I(data_in$dti*data_in$installment)
  
  data_in$intr3 <- I(data_in$emp_length*data_in$installment)
  data_in$intr4 <- I(data_in$last_pymnt_amnt*data_in$dti)
  data_in$intr5 <- I(data_in$last_pymnt_amnt*data_in$installment)
  data_in$intr6 <- I(data_in$last_pymnt_amnt*data_in$emp_length)
  
  data_in$intr7 <- I(data_in$loan_amnt*data_in$installment)
  data_in$intr8 <- I(data_in$loan_amnt*data_in$dti)
  data_in$intr9 <- I(data_in$loan_amnt*data_in$emp_length)
  data_in$intr10 <- I(data_in$loan_amnt*data_in$last_pymnt_amnt)
  
  data_in$intr11 <- I(data_in$total_pymnt*data_in$installment)
  data_in$intr12 <- I(data_in$total_pymnt*data_in$dti)
  data_in$intr13 <- I(data_in$total_pymnt*data_in$emp_length)
  data_in$intr14 <- I(data_in$total_pymnt*data_in$last_pymnt_amnt)
  data_in$intr15 <- I(data_in$total_pymnt*data_in$loan_amnt)
  
  data_in$intr16 <- I(data_in$total_rec_late_fee*data_in$installment)
  data_in$intr17 <- I(data_in$total_rec_late_fee*data_in$dti)
  data_in$intr18 <- I(data_in$total_rec_late_fee*data_in$emp_length)
  data_in$intr19 <- I(data_in$total_rec_late_fee*data_in$last_pymnt_amnt)
  data_in$intr20 <- I(data_in$total_rec_late_fee*data_in$loan_amnt)
  data_in$intr21 <- I(data_in$total_rec_late_fee*data_in$total_pymnt)
  
  #print (head(data_in))
  return (data_in)
}


###############   Using Polynomial models   #################

prep_polynomial <- function(data_in){
  data_in$p1 <- I(data_in$dti*data_in$dti)
  data_in$p2 <- I(data_in$installment*data_in$installment)
  data_in$p3 <- I(data_in$last_pymnt_amnt*data_in$last_pymnt_amnt)
  data_in$p4 <- I(data_in$loan_amnt*data_in$loan_amnt)
  data_in$p5 <- I(data_in$total_pymnt*data_in$total_pymnt)
  data_in$p6 <- I(data_in$emp_length*data_in$emp_length)
  data_in$p7 <- I(data_in$total_rec_late_fee*data_in$total_rec_late_fee)
  #print (head(data_in))
  return (data_in)
}


##############  Building Prediction vector  #################

prediction_vector <-function (output_in){
  cnt0 <- 0
  cnt1 <- 1
  pred <- vector()
  for (a in output_in){
    if (a<0.5) {
      cnt0 <- cnt0 + 1
      #append(pred, 0)
      pred <- c(pred, 0)
    } else{
      cnt1 <- cnt1 + 1
      #append(pred, 1)
      pred <- c(pred, 1)
    }
    #print (b)
  }
  return (pred)
}