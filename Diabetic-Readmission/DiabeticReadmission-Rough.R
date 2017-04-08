# Import libraries here
rm(list = ls())

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
library(repr)
library(glmnet)
library('caret')


# Get the data
df_Base = '/Users/sam/All-Program/App-DataSet/Spark-Operations/diabetes_dataset/df_Base.csv/'
df_F1 = '/Users/sam/All-Program/App-DataSet/Spark-Operations/diabetes_dataset/df_F1.csv/'
df_F2 = '/Users/sam/All-Program/App-DataSet/Spark-Operations/diabetes_dataset/df_F2.csv/'

df_Diag2Diag2NaN_rmv_F1 = '/Users/sam/All-Program/App-DataSet/Spark-Operations/diabetes_dataset/df_Diag1Diag2NaN_rmv_F1.csv/'
df_Diag2Diag2NaN_rmv_F2 = '/Users/sam/All-Program/App-DataSet/Spark-Operations/diabetes_dataset/df_Diag1Diag2NaN_rmv_F2.csv/'


runningDataSet <- df_F2

##################################################################################################

# Check the data if its looks alright
file_names <- list.files(path = runningDataSet, pattern = ".csv", full.names = TRUE)
data_DF <- do.call(rbind,lapply(file_names, read.csv, header=TRUE))
data_DF = data.frame(data_DF)
# Check how many unique output labels we have, should be binomial (no, and yes)
unique(data_DF$readmitted)
head(data_DF)

##################################################################################################

getData <- function(dataIN, featureType){
  # Do some small sanity check to be sure your data looks the same way as dumped before.
  # Check for numeric and factor columns and maatch them, if the are proper
  numCol <- length(names(dataIN))
  print ('The Total number of featurer incluing the Label and ID columns are: ') 
  numCol
  
  
  # Check if the correct features are numeric, if not then convert them to numeric and vice-a-versa 
  # c('time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses')
  print ('All Nueric features interpreted by R are: ') 
  names(which(sapply(dataIN, is.numeric)))
  
  # Check if the correct features are nominal, if not then convert them to factors and vice-a-versa
  print ('All Factor features interpreted by R are: ') 
  names(which(sapply(dataIN, is.factor)))
  
  # Converting the non-numeric columns into factors
  dataIN$admission_type_id <- as.factor(as.character(dataIN$admission_type_id))
  dataIN$discharge_disposition_id <- as.factor(as.character(dataIN$discharge_disposition_id))
  
  if (featureType == 'F1'){
    dataIN$admission_source_id <- as.factor(as.character(dataIN$admission_source_id))
  }
  
  # Converting readmitted into 1 and 0 where 1=yes and 0=no
  dataIN$readmitted<- as.factor(ifelse(dataIN$readmitted=='no', 0, 1))
  
  # Get the Numeric anf Factor features 
  numericCols <- names(which(sapply(dataIN, is.numeric)))
  # numericCols <- numericCols[numericCols != 'id']              # Removing the ID column
  factorCols <- names(which(sapply(dataIN, is.factor)))
  
  return (list(numericCols, factorCols, dataIN)) 
}


data_OUT <- getData(data_DF, 'F2')

print ('All numeric columns after DataType Conversion are: ')
numericCols <- data_OUT[[1]]
numericCols
print ('All factor columns after DataType Conversion are: ')
factorCols <- data_OUT[[2]]
factorCols
data_DFclean <- data_OUT[[3]]

# data_DF$readmitted<- as.factor(ifelse(data_DF$readmitted=="yes", 1, 0))
# Check if the 'yes' and 'no' of the readmitted columns are converted properly, should produce 0's and 1's
unique(data_DFclean$readmitted)
head(data_DFclean)


##################################################################################################

########## Common Functions  ############

bindModel <- function(yLabel, xFeatures=NULL, featureSet=NULL){
  # Automates the creation of feature model to be passed into an Classifier or Predictive Model
  if (is.null(xFeatures)){
    xFeatures <- featureSet[featureSet != yLabel]
  }   
  return (as.formula(paste(yLabel, "~", paste(xFeatures, collapse = '+ '))))
}

factorToDummy_DF_Builder <- function (dataFrameIN, numericCols, factorCols, yLabel){
  # Creates a design matrix by expanding the factors to a set of dummy variables and interaction etc.
  xNumeric <- dataFrameIN[, numericCols]
  xFactor <- dataFrameIN[, factorCols]
  
  factorModel <- bindModel(yLabel=yLabel, featureSet=factorCols)
  xFactor <- model.matrix(factorModel, data=xFactor)[, -1]        # -1 is provided to exclude the intercept term from the matrix
  readmitted <- dataFrameIN[[yLabel]]
  return (data.frame(xNumeric, xFactor, readmitted))
}


stratifiedSampling <- function(dataIN, sample_on_col, trainPrcnt, seed){
  set.seed(seed)
  trainIndices <- createDataPartition(y=dataIN[[sample_on_col]], p=trainPrcnt, list=FALSE)
  trainData <- dataIN[trainIndices,]
  testData <- dataIN[-trainIndices,]
  
  stopifnot(nrow(trainData) + nrow(testData) == nrow(dataIN))
  return (list(trainData, testData))
}


##################################################################################################

# Build the design matrix by expanding the factor variables into dummy variables
# We remove the columns 'diag_1', 'diag_2', 'diag_3' becasue they have any different categories and take much time to operate
excludVariables <- c('diag_1', 'diag_2', 'diag_3')
numericFeatures <- numericCols
nominalFeatures <- factorCols
nominalFeatures <- setdiff(nominalFeatures, excludVariables)
nominalFeatures

dataDF_dummy <- factorToDummy_DF_Builder(dataFrameIN=data_DFclean, numericCols=numericFeatures, factorCols=nominalFeatures, yLabel='readmitted')
head(dataDF_dummy)
print ('The number of instances are: ')
nrow(dataDF_dummy)
print ('The number of columns are; ')
ncol(dataDF_dummy)
# label <- dataMatrix$readmitted
# as.matrix


##################################################################################################
###################### Stratified Sampling #######################

# Stratified Sampling
dataOut <- stratifiedSampling(dataIN=dataDF_dummy, sample_on_col='readmitted', trainPrcnt=0.8, seed=45223)
trainData <- dataOut[[1]]
testData <- dataOut[[2]]
print ('Total number of training points are: ')
print (nrow(trainData))
print ('Total number of test points are: ')
print (nrow(testData))
# Convert teh dataframes into matrix of dummyDF to feed into the algorithm
# trainData <- as.matrix(trainData)
# testData <- as.matrix(testData)
# dataMatrix$readmitted
head(trainData)
head(testData)



##################################################################################################
###################### Common functions 2 #######################

getSummaryElements_glm <- function(model, alpha){
  inputSummary <- summary(model)
  inputSummary_DF <- as.data.frame(inputSummary$coefficients)  # z is for glm
  inputSummary_DF = inputSummary_DF[-1,]   # Remove the summary for intercept
  print (inputSummary_DF['Pr(>|z|)'])
  signFeatureIndice <- which(inputSummary_DF['Pr(>|z|)'] <= alpha)
  
  numFeatures <- nrow(inputSummary_DF) # -1 for the intercept
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


glmStats <- function(dataIN, max_p_val=0.20){
  cols <- colnames(dataIN)
  cols <- cols[1:length(cols)-1]   # removing readitted (the label from the list)
  print ('The list of columns are: ')
  print (cols)
  
  # stats <- c('numCategories', 'numSignCategories', 'signF1','signF2', 'signF3', 'nullDeviance', 'residualDeviance', 'aic', 'mean(resid)','min(resid)','q1(resid)','median(resid)','q3(resid)','max(resid)')
  # dataOut <- data.frame(matrix(ncol = length(stats), nrow = length(cols)))
  # colnames(dataOut) <- stats
  # rownames(dataOut) <- cols
  # 
  # # Change the columns signF1, signF2 and signF3 into characters
  # dataOut$signF1 <- as.character(dataOut$signF1)
  # dataOut$signF2 <- as.character(dataOut$signF2)
  # dataOut$signF3 <- as.character(dataOut$signF3)
  print ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
  dataOut <- data.frame()
  for (col in cols){
    print (col)
    modelF1.model <- bindModel(yLabel='readmitted', xFeatures=col)
    modelF1.glm <- glm(modelF1.model, family='binomial',data=dataIN)
    sumry <- getSummaryElements_glm(modelF1.glm, alpha=max_p_val)
    # print ('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
    # Feature Statistics
    dataOut[col,'numCategories'] <- sumry[["numFeatures"]]
    dataOut[col,'numSignCategories'] <- sumry[["numSignFeatures"]]
    dataOut[col,'signF1'] <- sumry[["topI_SignFeatures"]][1]
    dataOut[col,'signF2'] <- sumry[["topI_SignFeatures"]][2]
    dataOut[col,'signF3'] <- sumry[["topI_SignFeatures"]][3]
    # print ('cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
    # Statistics
    dataOut[col,'nullDeviance'] <- sumry[['nullDeviance']]
    dataOut[col,'residualDeviance'] <- sumry[['residualDeviance']]
    dataOut[col,'aic'] <- sumry[['aic']]
    # print ('dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd')
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

inpCols <- c("medical_specialtySurgery_Pediatric", "medical_specialtySurgery_Plastic", "medical_specialtySurgery_PlasticwithinHeadandNeck","readmitted")
head(trainData)
trainDataNew <- subset(trainData, select = inpCols)
head(trainDataNew)
stats.glm <- glmStats(trainDataNew, max_p_val=0.10)   # Note for most of the cases we exclude diag_1, diag_2, diag_3
stats.glm
unique(trainDataNew$medical_specialtySurgery_PlasticwithinHeadandNeck)
length(unique(trainDataNew$medical_specialtySurgery_Pediatric))
length(a)
##################################################################################################

unique()
stopifnot(colnames(trainData) == colnames(testData))
# colnames(dataF1_DF)[colSums(is.na(dataF1_DF)) > 0]

colnames()
head(dataF1_DF)

for (i in colnames(dataF1_DF)){
  print (i)
  print (unique(dataF1_DF[[i]]))
}

modelF1.model <- bindModel(yLabel='readmitted', featureSet=names(trainData))
modelF1.model
modelF1.glm <- glm(modelF1.model, family = binomial, data=dataF1_DF)

head(dataF1_DF$readmitted)
