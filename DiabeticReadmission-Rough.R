# Import libraries here
library(dplyr)
library(class)
library(MASS)
library(Hmisc)
library(klaR)
library(e1071)
library(kknn)
library(rpart)
library(lars)
library(stats)
library(leaps)
# library(repr)
library(glmnet)
library('caret')


# Get the data
df_Base = '/Users/sam/All-Program/App-DataSet/Spark-Operations/diabetes_dataset/df_Base.csv/'
df_F1 = '/Users/sam/All-Program/App-DataSet/Spark-Operations/diabetes_dataset/df_F1.csv/'
df_F2 = '/Users/sam/All-Program/App-DataSet/Spark-Operations/diabetes_dataset/df_F2.csv/'

df_Diag1Diag2NaN_rmv_F1 = '/Users/sam/All-Program/App-DataSet/Spark-Operations/diabetes_dataset/df_Diag1Diag2NaN_rmv_F1.csv/'
df_Diag1Diag2NaN_rmv_F2 = '/Users/sam/All-Program/App-DataSet/Spark-Operations/diabetes_dataset/df_Diag1Diag2NaN_rmv_F2.csv/'


file_names <- list.files(path = df_Diag1Diag2NaN_rmv_F1, pattern = ".csv", full.names = TRUE)
dataF1_DF <- do.call(rbind,lapply(file_names, read.csv, header=TRUE))


dataF1_DF = data.frame(dataF1_DF)
head(dataF1_DF)


##################################################################################################

# Do some small sanity check to be sure your data looks the same way as dumped before.


# Check for numeric and factor columns and maatch them, if the are proper
numCol <- length(names(dataF1_DF))
print ('The Total number of featurer incluing the Label and ID columns are: ') 
numCol


# Check if the correct features are numeric, if not then convert them to numeric and vice-a-versa 
# c('time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses')
print ('All Nueric features interpreted by R are: ') 
names(which(sapply(dataF1_DF, is.numeric)))

# Check if the correct features are nominal, if not then convert them to factors and vice-a-versa
print ('All Factor features interpreted by R are: ') 
names(which(sapply(dataF1_DF, is.factor)))

# Converting the non-numeric columns into factors
dataF1_DF$admission_type_id <- as.factor(as.character(dataF1_DF$admission_type_id))
dataF1_DF$discharge_disposition_id <- as.factor(as.character(dataF1_DF$discharge_disposition_id))
dataF1_DF$admission_source_id <- as.factor(as.character(dataF1_DF$admission_source_id))

# Converting readmitted into 1 and 0 where 1=yes and 0=no
dataF1_DF$readmitted<- as.factor(ifelse(dataF1_DF$readmitted=="yes", 1, 0))

# Get the Numeric anf Factor features 
numericCols <- names(which(sapply(dataF1_DF, is.numeric)))
# numericCols <- numericCols[numericCols != 'id']              # Removind the ID column
factorCols <- names(which(sapply(dataF1_DF, is.factor)))

# Samity check
print ('All numeric columns after DataType Conversion are: ')
numericCols
print ('All factor columns after DataType Conversion are: ')
factorCols


##################################################################################################


bindModel <- function(yLabel, xFeatures=NULL, featureSet=NULL){
  # Automates the creation of feature model to be passed into an Classifier or Predictive Model
  if (is.null(xFeatures)){
    xFeatures <- featureSet[featureSet != yLabel]
  }   
  return (as.formula(paste(yLabel, "~", paste(xFeatures, collapse = '+ '))))
}

factorMatrixBuilder <- function (dataFrameIN, numericCols, factorCols, yLabel){
  # Creates a design matrix by expanding the factors to a set of dummy variables and interaction etc.
  xNumeric <- dataFrameIN[, numericCols]
  xFactor <- dataFrameIN[, factorCols]
  
  factorModel <- bindModel(yLabel=yLabel, featureSet=factorCols)
  xFactor <- model.matrix(factorModel, data=xFactor)[, -1]        # -1 is provided to exclude the intercept term from the matrix
  
  return (as.matrix(data.frame(xNumeric, xFactor)))
}

dim(dataF1_DF)

stratifiedSampling <- function(dataIN, sample_on_col, trainPrcnt, seed){
  set.seed(seed)
  print (dim(dataIN[sample_on_col]))
  print (unique(dataIN[sample_on_col]))
  trainIndices <- createDataPartition(y=dataIN[[sample_on_col]], p=trainPrcnt, list=FALSE)
  trainData <- dataIN[trainIndices,]
  testData <- dataIN[-trainIndices,]
  print (dim(trainData))
  print (dim(testData))
  
  stopifnot(nrow(trainData) + nrow(testData) == nrow(dataIN))
  return (list(trainData, testData))
}


##################################################################################################

dataOut <- stratifiedSampling(dataIN=dataF1_DF, sample_on_col='readmitted', trainPrcnt=0.8, seed=45223)

trainData <- dataOut[[1]]
testData <- dataOut[[2]]

dim(trainData)
dim(testData)


##################################################################################################



for (i in colnames(dataDF)){
  print (i)
  print (unique(dataDF[[i]]))
}


getSummaryElements_glm <- function(model, alpha){
  inputSummary <- summary(model)
  inputSummary_DF <- as.data.frame(inputSummary$coefficients)  # z is for glm
  inputSummary_DF = inputSummary_DF[-1,]   # Remove the summary for intercept
  signFeatureIndice <- which(inputSummary_DF['Pr(>|z|)'] <= alpha)
  
  numFeatures <- nrow(inputSummary_DF) -1 # -1 for the intercept
  numSignFeatures <- length(signFeatureIndice)

  # Get the top i significant features
  i <- 1
  topI_SignFeatures <- 0
  while (i<=3){
    
    if (nrow(inputSummary_DF) >= 1){    # only if the summary dataframe has elements
      # Remove the entry of the intercept term
      min_index <- which(inputSummary_DF['Pr(>|z|)'] == min(inputSummary_DF['Pr(>|z|)']) & inputSummary_DF['Pr(>|z|)'] <= alpha)
      min_index
      if (length(min_index)!=0){
        topI_SignFeatures[i] <- rownames(inputSummary_DF)[min_index]
        inputSummary_DF = inputSummary_DF[-min_index,]
      } else {
        topI_SignFeatures[i] <- NaN
      }
    } else {
      topI_SignFeatures[i] <- NaN
    }
    
    i <- i+1
  }

  
  # print ('uoiweuifouweiofiowfoiof')
  # signFeatures <- row.names(inputSummary_DF[signFeatureIndice,])
  nullDeviance <- inputSummary$null
  residualDeviance <- inputSummary$deviance
  aic <- inputSummary$aic
  
  ## Residual Distribution:
  residuals <- resid(model)
  mean <- mean(residuals)
  min <- quantile(residuals)[['0%']]
  q1 <- quantile(residuals)[['25%']]
  median <- quantile(residuals)[['50%']]
  q3 <- quantile(residuals)[['75%']]
  max <- quantile(residuals)[['100%']]
  
  return (list(numFeatures=numFeatures, 
               numSignFeatures=numSignFeatures,
               topI_SignFeatures=topI_SignFeatures,
               nullDeviance=nullDeviance,
               residualDeviance=residualDeviance,
               aic=aic,
               resid_mean=mean,
               resid_min=min,
               resid_q1=q1,
               resid_median=median,
               resid_q3=q3,
               resid_max=max))
}


# inputSummary_DF <- as.data.frame(summary(modelF1.glm)$coefficients)['Pr(>|z|)']



glmStats <- function(dataIN, max_p_val=0.20){
  cols <- colnames(dataIN)
  cols <- cols[1:length(cols)-1]   # removing readitted (the label from the list)
  
  # stats <- c('numCategories', 'numSignCategories', 'signF1','signF2', 'signF3', 'nullDeviance', 'residualDeviance', 'aic', 'mean(resid)','min(resid)','q1(resid)','median(resid)','q3(resid)','max(resid)')
  # dataOut <- data.frame(matrix(ncol = length(stats), nrow = length(cols)))
  # colnames(dataOut) <- stats
  # rownames(dataOut) <- cols
  # 
  # # Change the columns signF1, signF2 and signF3 into characters
  # dataOut$signF1 <- as.character(dataOut$signF1)
  # dataOut$signF2 <- as.character(dataOut$signF2)
  # dataOut$signF3 <- as.character(dataOut$signF3)

  dataOut <- data.frame()
  for (col in cols){
    
    # print ('#####################################')
    # print (col)
    #     print (col)
    modelF1.model <- bindModel(yLabel='readmitted', xFeatures=col)
    #     print (modelF1.model)
    modelF1.glm <- glm(modelF1.model, family='binomial',data=dataIN)
    # print ('popopopopopopop')
    sumry <- getSummaryElements_glm(modelF1.glm, alpha=max_p_val)
    # print ('iuiuiuiuiuuiuiiu')
    # Feature Statistics
    dataOut[col,'numCategories'] <- sumry[["numFeatures"]]
    dataOut[col,'numSignCategories'] <- sumry[["numSignFeatures"]]
    dataOut[col,'signF1'] <- sumry[["topI_SignFeatures"]][1]
    dataOut[col,'signF2'] <- sumry[["topI_SignFeatures"]][2]
    dataOut[col,'signF3'] <- sumry[["topI_SignFeatures"]][3]
    
    # Statistics
    dataOut[col,'nullDeviance'] <- sumry[['nullDeviance']]
    dataOut[col,'residualDeviance'] <- sumry[['residualDeviance']]
    dataOut[col,'aic'] <- sumry[['aic']]
    
    # Residual Statistics
    dataOut[col,'mean(resid)'] <- sumry[["resid_mean"]]
    dataOut[col,'min(resid)'] <- sumry[["resid_min"]]
    dataOut[col,'q1(resid)'] <- sumry[['resid_q1']]
    dataOut[col,'median(resid)'] <- sumry[['resid_median']]
    dataOut[col,'q3(resid)'] <- sumry[['resid_q3']]
    dataOut[col,'max(resid)'] <- sumry[['resid_max']]
    
    #     print ('################################################################################')
    #     print ('')
  }
  return (dataOut)
}


dataDF <- data.frame(subset(trainData, select = -c(diag_1,diag_2,diag_3)))
dataDF <- subset(trainData, select = c(age,race,time_in_hospital, readmitted))
stats.glm <- glmStats(dataDF)


#######  Check

cols <- colnames(dataDF)
stats <- c('numCategories', 'numSignCategories', 'signF1','signF2', 'signF3', 'nullDeviance', 'residualDeviance', 'aic', 'mean(resid)','min(resid)','q1(resid)','median(resid)','q3(resid)','max(resid)')
dataOut <- data.frame(matrix(ncol = length(stats), nrow = length(cols)), stringsAsFactors=FALSE)
colnames(dataOut) <- stats
rownames(dataOut) <- cols
dataOut <- data.frame()

col <- 'time_in_hospital'
modelF1.glm <- glm(readmitted~time_in_hospital, family='binomial',data=dataDF)
sumry <- getSummaryElements_glm(modelF1.glm, alpha=0.20)

dataOut[col,'numCategories'] <- sumry[["numFeatures"]]
dataOut[col,'numSignCategories'] <- sumry[["numSignFeatures"]]
dataOut[col,'signF1'] <- sumry[["topI_SignFeatures"]][1]
dataOut[col,'signF2'] <- sumry[["topI_SignFeatures"]][2]
dataOut[col,'signF3'] <- sumry[["topI_SignFeatures"]][3]

# Statistics
dataOut[col,'nullDeviance'] <- sumry[['nullDeviance']]
dataOut[col,'residualDeviance'] <- sumry[['residualDeviance']]
dataOut[col,'aic'] <- sumry[['aic']]

# Residual Statistics
dataOut[col,'mean(resid)'] <- sumry[["resid_mean"]]
dataOut[col,'min(resid)'] <- sumry[["resid_min"]]
dataOut[col,'q1(resid)'] <- sumry[['resid_q1']]
dataOut[col,'median(resid)'] <- sumry[['resid_median']]
dataOut[col,'q3(resid)'] <- sumry[['resid_q3']]
dataOut[col,'max(resid)'] <- sumry[['resid_max']]







############################################################################################################

col = 'time_in_hospital'
modelF1.glm <- glm(readmitted~time_in_hospital, family='binomial',data=dataDF)
inputSummary <- summary(modelF1.glm)
inputSummary


inputSummary_DF <- as.data.frame(inputSummary$coefficients)  # z is for glm
inputSummary_DF
inputSummary_DF = inputSummary_DF[-1,] 
signFeatureIndice <- which(inputSummary_DF['Pr(>|z|)'] <= 0.20)
signFeatureIndice
# signFeatureIndice <- signFeatureIndice[signFeatureIndice!=1]
# signFeatureIndice

numSignFeatures <- length(signFeatureIndice)
numSignFeatures
# inputSummary_DF = inputSummary_DF[-1,] 
i <- 1
signF1 <- 0
while (i<=3){
  # Remove the entry of the intercept term
  if (nrow(inputSummary_DF) >= 1){
    min_index <- which(inputSummary_DF['Pr(>|z|)'] == min(inputSummary_DF['Pr(>|z|)']) & inputSummary_DF['Pr(>|z|)'] <= 0.20)
    min_index
    if (length(min_index)!=0){
      signF1[i] <- rownames(inputSummary_DF)[min_index]
      inputSummary_DF = inputSummary_DF[-min_index,]
    } else {
      signF1[i] <- 0
    }
  }
  else{
    signF1[i] <- 0
  }
  i <- i+1
}

signF1[1]

nrow(inputSummary_DF)==0


inputSummary_DF = inputSummary_DF[-1,]   # Remove the entry of the intercept term
min_index <- which(inputSummary_DF['Pr(>|z|)'] == min(inputSummary_DF['Pr(>|z|)']))
min_index
signFeature1 <- rownames(inputSummary_DF)[min_index]

inputSummary_DF = inputSummary_DF[-min_index,]

signFeatureIndice <- which(inputSummary_DF['Pr(>|z|)'] <= alpha)
signFeatureIndice <- signFeatureIndice[signFeatureIndice!=1]   # remo





# numFeatures <- nrow(inputSummary_DF) -1 # -1 for the intercept
# numFeatures
# signFeatures <- row.names(inputSummary_DF[signFeatureIndice,])
# signFeatures
nullDeviance <- inputSummary$null
nullDeviance
residualDeviance <- inputSummary$deviance
residualDeviance
aic <- inputSummary$aic
aic


residuals <- resid(modelF1.glm)
mean <- mean(residuals)
mean
min <- quantile(residuals)[['0%']]
min
q1 <- quantile(residuals)[['25%']]
q1
median <- quantile(residuals)[['50%']]
median
q3 <- quantile(residuals)[['75%']]
q3
max <- quantile(residuals)[['100%']]
max




min(resid(modelF1.glm))
mean(resid(modelF1.glm))
median(resid(modelF1.glm))
max(resid(modelF1.glm))
q1(resid(modelF1.glm))

sumry <- getSummaryElements_glm(summary(modelF1.glm), alpha=0.20)