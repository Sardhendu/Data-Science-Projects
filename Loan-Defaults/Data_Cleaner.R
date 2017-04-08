

# turn a date into a 'monthnumber' relative to an origin
monnb <- function(d) 
{ 
  lt <- as.POSIXlt(as.Date(paste("01-", d, sep = ""), "%d-%b-%Y"))
  return (lt$year*12 + lt$mon) 

}

# compute a month difference as a difference between two monnb's
mondf <- function(d1, d2) 
{ 
   return (monnb(d2) - monnb(d1)) 
}


clean <- function(data_in)
{
  #Interest-Rate Convert from factor to string, replace % and convert to numeric
  data_in$int_rate <- as.numeric(sub("%","",sapply(data_in$int_rate, as.character)))
  #data_in$int_rate <- NULL
  
  #Enploylength -- Keep only the digits and convert to digit
  data_in$emp_length <- as.numeric(gsub("\\D","",sapply(data_in$emp_length,as.character)))
  #data_in$emp_length <- NULL
  
  # Zip code replacing xx
  #print (data_in$zip_code)
  data_in$zip_code <- as.character(sub("xx","",sapply(data_in$zip_code,as.character)))
  #data_in$zip_code <- NULL
  
  # Revol_util -- Replace % and convert to float
  data_in$revol_util <- as.numeric(sub("%","",sapply(data_in$revol_util, as.character)))
  #data_in$revol_util <- NULL
  
  # Date Transformation
  data_in$earliest_cr_line <- mondf(data_in$earliest_cr_line, "Dec-2016")
  
  # Remove all the columns that have more than 85% of NA
  col_names <- names(data_in)[colSums(is.na(data_in)) >= 0.85 * dim(data_in)[1]]

  return (drop_columns(data_in, col_names))
}


drop_columns <- function(data_in, col_names)
{
  for (cn in col_names) {
    data_in[,paste(cn)]<-NULL
  }
  return(data_in)
}


var_cleaner = function(data_in)
{
  data_in$loan_status.new <- data_in$loan_status == 'Fully Paid'
  data_in['loan_status.new'] <- lapply(data_in['loan_status.new'], as.integer)
  
  #data_in$annual_inc <- as.numeric(as.character(data_in$annual_inc))
  data_in$dti <- as.numeric(as.character(data_in$dti))
  data_in$installment <- as.numeric(as.character(data_in$installment))
  data_in$last_pymnt_amnt <- as.numeric(as.character(data_in$last_pymnt_amnt))
  data_in$installment <- as.numeric(as.character(data_in$installment))
  #data_in$pymnt_plan <- as.numeric(as.character(data_in$pymnt_plan))
  data_in$out_prncp <- as.numeric(as.character(data_in$out_prncp))
  data_in$out_prncp_inv <- as.numeric(as.character(data_in$out_prncp_inv))
  #data_in$recoveries <- as.numeric(as.character(data_in$recoveries))
  data_in$total_pymnt <- as.numeric(as.character(data_in$total_pymnt))
  #data_in$total_pymnt_inv <- as.numeric(as.character(data_in$total_pymnt_inv))
  data_in$total_rec_int <- as.numeric(as.character(data_in$total_rec_int))
  data_in$total_rec_late_fee <- as.numeric(as.character(data_in$total_rec_late_fee))
  data_in$total_rec_prncp <- as.numeric(as.character(data_in$total_rec_prncp))
  return(data_in)
}


##################   Handle Missing Values   ###################
delete.na <- function(DF, n=0) {
  DF[rowSums(is.na(DF)) <= n,]
  return (DF)
}

na_rplc_median <- function(DF){
  NA_col <- colnames(DF)[colSums(is.na(DF)) > 0]
  for (col_name in NA_col){
    col_median <- summary(DF[[col_name]])['Median']
    DF[col_name][is.na(DF[col_name])] <- col_median
  }
  return (DF)
}




###########   Standarize the dataset with average   ###########
avg_standarize <- function (data_tot, data_in){
  columns <- colnames(data_tot)
  for (col_names in colnames(data_tot)){
    # print (col_names)
    # print (1111111111)
    col_dtype <- (unique(sapply(data_tot[[col_names]], class)))
    if (col_dtype=="integer" || col_dtype=="numeric"){
      mean = mean(data_tot[[col_names]])
      if (mean != 0){
        print (col_names)
        print (mean)
        data_in[[col_names]]  <- data_in[[col_names]] * (1/mean)
      }
    }
  }
  return (data_in)
}

