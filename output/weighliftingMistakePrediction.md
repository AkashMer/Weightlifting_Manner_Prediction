---
title: "Prediction of Weightlifting mistakes based on sensor data"
author: "Akash Mer"
date: "2023-07-21"
output: 
  html_document:
    keep_md: true
    code_folding: hide
    theme: readable
    highlight: espresso
    toc: yes
    toc_float:
        collapsed: true
knit: (function(input, ...){
    rmarkdown::render(input,
        output_dir = "C:/Users/akash/Documents/datasciencecoursera/Weightlifting_Manner_Prediction/output",
        output_file = file.path("./weighliftingMistakePrediction"))
    })
---

# **INTRODUCTION**
  
This report explores the data obtained from multiple sensors attached to 4 different places while doing a Unilateral Dumbbell Biceps Curl in 6 different subjects. The primary goal is to train a machine learning model and use it to predict specified execution and common mistakes.  
**Mistakes predicted** : 
  
* *throwing elbows to the front*
* *lifting dumbbell only halfway*
* *lowering the dumbbell only halfway*
* *throwing hips to the front*
  
**Sensor placements** :
  
* *waist belt* - on the back side
* *arm-band* - tied around arm
* *glove* - sensor placed close to forearm
* *dumbbell* - on the inner side of the dumbbell
  
**Reference** : The data was provided by the instructors of the Practical Machine Learning Course as part of  a course project  
Further details on the experiment can be found [here](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har)  
[1] Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. [Qualitative Activity Recognition of Weight](http://web.archive.org/web/20161224072740/http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf) Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
  
# **DATA PROCESSING**
  
1. **Reading in the training data** : The training data provided for this report can be found [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv). The training data was read in and stored as *trainDat* data frame object

```r
trainingDat <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
                        header = TRUE, na.strings = c("","NA"))
```
2. **Rading in the data to be predicted** : The data for which the outcome is to be predicted can be found [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv). This data was read in and stored as *datToPredicted* data frame object

```r
datToPredict <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                        header = TRUE, na.strings = c("","NA"))
```
3. **Codebook** : The official code book for the data is not available but this code book provides the best guess by referencing the research paper mentioned above
    + **X** : Row number
    + **user_name** : Subject identifier
    + **raw_timestamp_part_1** : Time when the measurement was taken represented as number of seconds from 1 January, 1970
    + **raw_timestamp_part_2** : Milliseconds to be further added to the part 1 to get the exact time of measurement
    + **cvtd_timestamp** : Time when the measurement was taken represented as date + time in hours and minutes
    + **new_window** : Marks(with "yes") the rows when current sliding window ends
    + **num_window** : Sliding window identifier
    + **Columns 8 to 159** : All of these have desciptive labels and represent data obtained from each sensors which include, raw *accelerometer, gyroscope and magnetometer* readings along the x,y and z axes, calculated *total and variance of acceleration*, calculated Euler angles(*roll, pitch, yaw*), and calculated *mean, variance, standard deviation, max, min, amplitude, kurtosis and skewness of the Euler angles*. Thus in total 38 features for each of the 4 sensors ie. **152 features** in total
    + **classe** : Weightlifting manner identifier labelled as A, B, C, D, and E - our **outcome variable**
4. **Data Partitition** : The training data set is partitioned with *seed set at 44725* into the following,
    + **train** set : 60%
    + **test** set : 20%
    + **validation** set : 20%

```r
if(system.file(package = "caret") == "") install.packages("caret")
if(system.file(package = "ggplot2") == "") install.packages("ggplot2")
if(system.file(package = "lattice") == "") install.packages("lattice")
suppressPackageStartupMessages(library(caret))
# Splitting out the train data set
set.seed(54546)
trainingIndex <- createDataPartition(trainingDat$classe, p = 0.6, list = FALSE)
train <- trainingDat[trainingIndex,]
nonTrain <- trainingDat[-trainingIndex,]
# Splitting this further to test and validation sets
set.seed(54546)
testIndex <- createDataPartition(nonTrain$classe, p = 0.5, list = FALSE)
test <- nonTrain[testIndex,]
validation <- nonTrain[-testIndex,]
```

```r
# Percentage of missing values in each column
percentNA <- sapply(1:ncol(train), function(i) mean(is.na(train[,i])))
NAcolumns <- which(percentNA != 0)
```
5. **Exploring missing values** : A closer look at the train set suggests a lot of missing values. This trend extends to the test, validation and the datToPredict data sets as well. Number of columns with missing values in the **train set is, 100** with each having the same percentage of missing values(**97.82%**). Further investigation shows these columns only have values when the current window is about to end. The characteristic pattern suggests they are missing values of the type **MAR**(*Missing at random* instead of MCAR or MNAR) since the likeliness of the data being missing can be clearly estimated from the other variables which have no missing data. Thus, with such a high percentage of missing values in each of these columns, it would not be prudent to try to impute them in some way since this would reduce the variability of our model and increase the bias towards the selected imputation method for eg. imputing with values for that corresponding window would cause these variables to have very low variability, also, all of these are calculated features so we would not be losing a lot of information by ignoring these variables since they were derived from the measured variables which have 0 missing values
6. **Removing columns with missing values** : Hence, ignoring these variables for the model building, the data set needs to be transformed to exclude these variables. Using the index generated only from train set, other data sets were also transformed. The *datToPredict* data set was stored in a new data set to preserve the original

```r
train <- train[,-NAcolumns]
test <- test[,-NAcolumns]
validation <- validation[,-NAcolumns]
datToPredictTransformed <- datToPredict[,-NAcolumns]
```
7. **Exploring the first 7 variables** : After the transformation for NAs, the data set had 60 variables and 100% complete cases. The first 7 columns were ignored as well due to the following reasons
    + **Row identifier[1]** - It is just a row identifier and does not affect the outcome
    + **Subject name[2]** - All users were of the same sex and similar age(20-28) and performed the activities under the same conditions with the same weighted dumbbell(1.25kg) under the supervision of an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate^1. Under such controlled conditions, any inter-subject variability was already controlled for in the experiment design
    + **Time stamps[3,4,5]** - These represent information on data collection times rather than any measurement related to the manner of activity since we are not trying to predict a forecast on these activities, but the mistakes while doing the activity
    + **Window information[6,7]** - These also represent information on data collection and would be more helpful if we were trying to predict which type of sliding window length performs better, which seems more of a question from a design perspective than the question we are trying to answer, which is to predict mistakes while weightlifting
8. **Removing the first 7 variables** : Hence, the first 7 columns were removed from all data sets

```r
# Removing first 7 variables
train <- train[,-(1:7)]
test <- test[,-(1:7)]
validation <- validation[,-(1:7)]
datToPredictTransformed <- datToPredictTransformed[,-(1:7)]
```
9. **Converting outcome to a factor with descriptive labels** : The outcome variable **classe** was converted to a factor variable with descriptive labels for each manner of activity and moved to the 1st column. Same transformations will be applied to all the other data sets, except for the *datToPredict* data set, where the outcome is unknown but the outcome column is replaced by the problem_id column, which will be retained and shifted to the first column

```r
if(system.file(package = "dplyr") == "") install.packages("dplyr")
suppressPackageStartupMessages(library(dplyr))
# Giving descriptive labels to the outcome variable
outcomes <- c("noMistake", "throwingElbowsFront", "liftingHalfway", "loweringHalfway",
              "throwingHipsFront")
# Retaining the old labels for future use
classeLabels <- unique(train$classe)
# Converting to factor variable and rearranging the variables
train <- train %>%
    mutate(outcome = factor(classe, labels = outcomes)) %>%
    select(outcome, everything(), -classe)
test <- test %>%
    mutate(outcome = factor(classe, labels = outcomes)) %>%
    select(outcome, everything(), -classe)
validation <- validation %>%
    mutate(outcome = factor(classe, labels = outcomes)) %>%
    select(outcome, everything(), -classe)
datToPredictTransformed <- datToPredictTransformed %>%
    select(problem_id, everything())
```
  
Thus, our tidy data sets have the following dimensions,
  
* **train** - 11776, 53
* **test** - 3923, 53
* **validation** - 3923, 53
* **datToPredictTransformed** - 20, 53
  
**Code book**
  
* **outcome** : Targeted outcome variable with the following levels - noMistake, throwingElbowsFront, liftingHalfway, loweringHalfway, throwingHipsFront
* **Columns to 2 to 53** : Features for our model. All of these have desciptive labels and represent data obtained from each sensors which include, raw *accelerometer, gyroscope and magnetometer* readings along the *x,y and z axes*, calculated *total acceleration*, calculated Euler angles(*roll, pitch, yaw*). Thus in total 13 features for each of the 4 sensors ie. **52 features** in total
* **problem_id**(only for *datToPredictTransformed* data set) : Identifier for submission of results of model predictions
  
# **EXPLORATORY DATA ANALYSIS**
  

```r
if(system.file(package = "GGally") == "") install.packages("GGally")
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(GGally))

# Converting our outcome variable to numeric just for the correlation plot
trainNum <- train
trainNum$outcome <- as.numeric(trainNum$outcome)
names(trainNum) <- 1:53

# Correlation plot
plotA <- ggcorr(trainNum, size = 10) +
    labs(title = "Plot A : Correlation plot for all variables in the data set",
         subtitle = "X1 = outcome, X2:X53 = features",
         caption = "Reference: Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H.\n
         Qualitative Activity Recognition of Weight Lifting Exercises\n
         Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13)\n
         Stuttgart, Germany: ACM SIGCHI, 2013") +
    theme_bw(base_size = 55) +
    theme(legend.key.size = unit(4, "cm"),
          legend.text = element_text(size = 55),
          plot.title = element_text(size = 70),
          plot.caption = element_text(size = 30, face = "italic"))

# Printing the plot
print(plotA)
```

![](C:/Users/akash/Documents/datasciencecoursera/Weightlifting_Manner_Prediction/output/weighliftingMistakePrediction_files/figure-html/correlation-1.png)<!-- -->
  
**Conclusions**
  
* High correlations among the following features - *roll_belt, pitch_belt, yaw_belt, total_accel_belt, gyros_belt_x, gyros_belt_y, gyros_belt_z, accel_belt_x, accel_belt_y, accel_belt_z, magnet_belt_x*. All of these are from the sensor on the belt, which is to be expected since these values would change in a similar direction while doing a bicep curl since the sensors only measure a change in values and not the absolute values. These features do show similar correlations with our outcome, so making a decision on which features to exclude and still retain all the information cannot be made. So all these features will be retained
* High correlations among the following features - *accel_arm_x, magnet_arm_x, magnet_arm_y, magnet_arm_z*. All of these are from the sensor on the arm, which again should see a similar change in values while doing a bicep curl. These features do show similar correlations with our outcome, so making a decision on which features to exclude and still retain all the information cannot be made. So all these features will be retained
* High correlations among the following features - *roll_dumbbell, pitch_dumbbell, yaw_dumbbell, accel_dumbbell_x, accel_dumbbell_y, accel_dumbbell_z*. All of these are from the sensor on the dumbbell, which again should see a similar change in values while doing a bicep curl. These features do show similar correlations with our outcome, so making a decision on which features to exclude and still retain all the information cannot be made. So all these features will be retained
  
Despite the above mentioned and a few more areas in *plotA*, *majority of features show low correlations with each other*, thus there are **low chances of bias due to multicollinearity**. Also, since the **primary goal of our model is prediction and not interpretation or selection of important features**, we will not be so concerned by bias from multicollinearity if it does occur. To **reduce the chances of it happening we will be using some sort of bootstrapping or cross validation methods for our models** and **use accuracy as a measure to compare models which is usually not so highly influenced by multicollinearity since it only concerns with predictions and truth in the population**
  
# **STATISTICAL MODELING**
  
## **Plan**
  
1. Build **at least 2 classification type models with cross validation** on the *train* set
2. Build **a maximum of 3 classification type models with cross validation** on the *train* set in case of lower accuracy of the first 2 models on the train set(< 90% in one of them)
3. **Predict and check the accuracy of the models on the test set**
4. **Build a combined model(by stacking the above models)** using **random forests with bootstrapping and controlling out-of-bag accuracy scores** on the test set
5. **Predict and check the accuracy of all the models on the validation set**
6. **Decide on the best model based on accuracy on validation set**
7. **Apply on best model to the datToPredictTransformed set** to predict the manner of weightlifting
  
## **Model building**
  
*Model 1* : **Gradient boosting with trees** - good choice based on the weak correlations of all the features to the outcome variable as seen in *plotA*. The **resampling method chosen is 5-fold repeated cross validation with 5 repeats**. Seed was set at *42684*

```r
if(system.file(package = "gbm") == "") install.packages("gbm")
suppressPackageStartupMessages(library(gbm))
# Model 1
set.seed(42684)
modelGBM <- train(outcome ~ ., data = train, method = "gbm", verbose = FALSE,
                  trControl = trainControl(method = "repeatedcv", number = 5,
                                           repeats = 5))
modelGBM
```

```
## Stochastic Gradient Boosting 
## 
## 11776 samples
##    52 predictor
##     5 classes: 'noMistake', 'throwingElbowsFront', 'liftingHalfway', 'loweringHalfway', 'throwingHipsFront' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold, repeated 5 times) 
## Summary of sample sizes: 9420, 9420, 9421, 9422, 9421, 9421, ... 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.7512401  0.6846617
##   1                  100      0.8188012  0.7706826
##   1                  150      0.8528532  0.8138107
##   2                   50      0.8531933  0.8140317
##   2                  100      0.9064707  0.8816462
##   2                  150      0.9314709  0.9132895
##   3                   50      0.8956181  0.8678823
##   3                  100      0.9404719  0.9246764
##   3                  150      0.9596125  0.9489027
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were n.trees = 150, interaction.depth =
##  3, shrinkage = 0.1 and n.minobsinnode = 10.
```
  
*Model 2* : **k-Nearest Neighbours** - A simple model which is easy to interpret and has no assumptions about the data and uses distances between the data to predict instead of weighting each predictor and add them up(characteristics of gradient boosting). Also kNN accounts for any non-linearity in the data. The **resampling method chosen is 10-fold cross-validation**. Seed was set at *42684*

```r
if(system.file(package = "kknn") == "") install.packages("kknn")
suppressPackageStartupMessages(library(kknn))
# Model 2
set.seed(42684)
modelkNN <- train(outcome ~ ., data = train, method = "kknn", verbose = FALSE,
                  trControl = trainControl(method = "cv", number = 10))
modelkNN
```

```
## k-Nearest Neighbors 
## 
## 11776 samples
##    52 predictor
##     5 classes: 'noMistake', 'throwingElbowsFront', 'liftingHalfway', 'loweringHalfway', 'throwingHipsFront' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 10599, 10598, 10599, 10598, 10599, 10598, ... 
## Resampling results across tuning parameters:
## 
##   kmax  Accuracy   Kappa    
##   5     0.9847993  0.9807761
##   7     0.9847993  0.9807761
##   9     0.9847993  0.9807761
## 
## Tuning parameter 'distance' was held constant at a value of 2
## Tuning
##  parameter 'kernel' was held constant at a value of optimal
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were kmax = 9, distance = 2 and kernel
##  = optimal.
```
  
Accuracy of more than 90% was achieved on both models, so a 3rd model will not be built

## **Predictions on the test set**
  

```r
if(system.file(package = "scales") == "") install.packages("scales")
suppressPackageStartupMessages(library(scales))
if(system.file(package = "gridExtra") == "") install.packages("gridExtra")
suppressPackageStartupMessages(library(gridExtra))
if(system.file(package = "cowplot") == "") install.packages("cowplot")
suppressPackageStartupMessages(library(cowplot))


# Predictions on the test set
predictionsGBM <- predict(modelGBM, test)
predictionskNN <- predict(modelkNN, test)

# Accuracy on the test set
confMatGBM <- confusionMatrix(predictionsGBM, test$outcome)
confMatkNN <- confusionMatrix(predictionskNN, test$outcome)

# Convert the confusion matrix into a percentage
# in each cell for better comparison
# Storing it as a data frame since we will be plotting using ggplot2
GBMtest <- as.data.frame(confMatGBM$table / rowSums(confMatGBM$table))
kNNtest <- as.data.frame(confMatkNN$table / rowSums(confMatkNN$table))

# Defining values for re-scaling of the gradient of the color on the plots
valueSeq <- c(seq(0,0.05,length.out = 200),seq(0.8,1,length.out = 500))

# Displaying the confusion matrix as plots
plotB <- ggplot(GBMtest, aes(Reference, Prediction, fill = Freq)) +
    geom_tile(alpha = 0.7) +
    geom_text(aes(label = percent(Freq, accuracy = 0.01)), size = 13) +
    scale_fill_distiller(type = "div", palette = "RdYlGn", direction = 1,
                         values = valueSeq) +
    labs(title = "Plot B : Confusion Matrix for the GBM Model") +
    theme_bw(base_size = 55) +
    theme(axis.text.x = element_text(angle = 20), legend.position = "none",
          plot.title = element_text(size = 70))

tableB <- tableGrob(data.frame(Percent = round(confMatGBM$overall[1:4]*100, 2)),
                    theme = ttheme_default(base_size = 50))

plotC <- ggplot(kNNtest, aes(Reference, Prediction, fill = Freq)) +
    geom_tile(alpha = 0.7) +
    geom_text(aes(label = percent(Freq, accuracy = 0.01)), size = 13) +
    scale_fill_distiller(type = "div", palette = "RdYlGn", direction = 1,
                         values = valueSeq) +
    labs(title = "Plot C : Confusion Matrix for the kNN Model") +
    theme_bw(base_size = 55) +
    theme(axis.text.x = element_text(angle = 20), legend.position = "none",
          plot.title = element_text(size = 70))

tableC <- tableGrob(data.frame(Percent = round(confMatkNN$overall[1:4]*100, 2)),
                    theme = ttheme_default(base_size = 50))

confusionMatrixPlots <- plot_grid(plotB, tableB, NULL, NULL, plotC, tableC, ncol = 2,
                                  rel_widths = c(2,1), rel_heights = c(1,0.05,1)) +
    labs(caption = "Reference: Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H.\n
         Qualitative Activity Recognition of Weight Lifting Exercises\n
         Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13)\n
         Stuttgart, Germany: ACM SIGCHI, 2013") +
    theme(plot.caption = element_text(face = "italic", size = 30, hjust = 1))

print(confusionMatrixPlots)
```

![](C:/Users/akash/Documents/datasciencecoursera/Weightlifting_Manner_Prediction/output/weighliftingMistakePrediction_files/figure-html/predictions-1.png)<!-- -->
  
Accuracy of more than 90% was also achieved on the test set

## **Agreement of both models on the test set**
  

```r
# Confusion matrix for the agreement between the models
confMatAgree <- confusionMatrix(predictionsGBM, predictionskNN)

# Convert the confusion matrix into a percentage
# in each cell for better comparison
# Storing it as a data frame since we will be plotting using ggplot2
agreementTest <- as.data.frame(confMatAgree$table / rowSums(confMatAgree$table))

# Displaying the confusion matrix as plots
plotD <- ggplot(GBMtest, aes(Reference, Prediction, fill = Freq)) +
    geom_tile(alpha = 0.7) +
    geom_text(aes(label = percent(Freq, accuracy = 0.01)), size = 13) +
    scale_fill_distiller(type = "div", palette = "RdYlGn", direction = 1,
                         values = valueSeq) +
    labs(title = "Plot D : Confusion Matrix for agreement between both models",
         x = "Predictions from kNN Model", y = "Predictions from the GBM Model") +
    theme_bw(base_size = 55) +
    theme(axis.text.x = element_text(angle = 20), legend.position = "none",
          plot.title = element_text(size = 70))

tableD <- tableGrob(data.frame(Agreement = round(confMatAgree$overall[c(1,3,4)]*100, 2)),
                    theme = ttheme_default(base_size = 50))

agreementPlots <- plot_grid(plotD, tableD, ncol = 2, rel_widths = c(2,1)) +
    labs(caption = "Reference: Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H.\n
         Qualitative Activity Recognition of Weight Lifting Exercises\n
         Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13)\n
         Stuttgart, Germany: ACM SIGCHI, 2013") +
    theme(plot.caption = element_text(face = "italic", size = 30, hjust = 1))

print(agreementPlots)
```

![](C:/Users/akash/Documents/datasciencecoursera/Weightlifting_Manner_Prediction/output/weighliftingMistakePrediction_files/figure-html/agreement-1.png)<!-- -->

Since both models do not completely agree with each other, a combined model was built using random forests by stacking both models

## **Combined Model**
  
A combined model by stacking the above models was built using **Random Forests** which uses the majority vote concept and in turn increase the accuracy. The **resampling method involves bootstrapping with oob(controlling out-of-bag accuracy scores) with 20 resamples**. Seed was set at *42684*.  
This combined model is built on the test set and then will be applied to the validation set for determining the accuracy  

```r
if(system.file(package = "randomForest") == "") install.packages("randomForest")
suppressPackageStartupMessages(library(randomForest))

# Stacking the predictions from both models
testPredictionsDF <- data.frame(predictionsGBM, predictionskNN, outcome = test$outcome)

# Model 2
set.seed(42684)
combModelRF <- train(outcome ~ ., data = testPredictionsDF, method = "rf",
                     verbose = FALSE,
                     trControl = trainControl(method = "oob", number = 20))
combModelRF
```

```
## Random Forest 
## 
## 3923 samples
##    2 predictor
##    5 classes: 'noMistake', 'throwingElbowsFront', 'liftingHalfway', 'loweringHalfway', 'throwingHipsFront' 
## 
## No pre-processing
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##   2     0.9841958  0.9800094
##   5     0.9841958  0.9800094
##   8     0.9849605  0.9809754
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 8.
```
  
Accuracy of 99% was achieved on the test set with the combined model which is higher than either models
  
## **Predictions on the validation set**
  

```r
library(ggplot2)

# Predictions on the validation set
predictionsGBM <- predict(modelGBM, validation)
predictionskNN <- predict(modelkNN, validation)
validationPredictionsDF <- data.frame(predictionsGBM, predictionskNN)
predictionsCombRF <- predict(combModelRF, validationPredictionsDF)

# Accuracy on the validation set
confMatGBM <- confusionMatrix(predictionsGBM, validation$outcome)
confMatkNN <- confusionMatrix(predictionskNN, validation$outcome)
confMatCombRF <- confusionMatrix(predictionsCombRF, validation$outcome)

# Convert the confusion matrix into a percentage
# in each cell for better comparison
# Storing it as a data frame since we will be plotting using ggplot2
GBMtest <- as.data.frame(confMatGBM$table / rowSums(confMatGBM$table))
kNNtest <- as.data.frame(confMatkNN$table / rowSums(confMatkNN$table))
combRFtest <- as.data.frame(confMatCombRF$table / rowSums(confMatCombRF$table))

# Defining values for re-scaling of the gradient of the color on the plots
valueSeq <- c(seq(0,0.05,length.out = 200),seq(0.8,1,length.out = 500))

# Displaying the confusion matrix as plots
plotE <- ggplot(GBMtest, aes(Reference, Prediction, fill = Freq)) +
    geom_tile(alpha = 0.7) +
    geom_text(aes(label = percent(Freq, accuracy = 0.01)), size = 13) +
    scale_fill_distiller(type = "div", palette = "RdYlGn", direction = 1,
                         values = valueSeq) +
    labs(title = "Plot E : Confusion Matrix for the GBM Model") +
    theme_bw(base_size = 55) +
    theme(axis.text.x = element_text(angle = 20), legend.position = "none",
          plot.title = element_text(size = 70))

tableE <- tableGrob(data.frame(Percent = round(confMatGBM$overall[1:4]*100, 2)),
                    theme = ttheme_default(base_size = 50))

plotF <- ggplot(kNNtest, aes(Reference, Prediction, fill = Freq)) +
    geom_tile(alpha = 0.7) +
    geom_text(aes(label = percent(Freq, accuracy = 0.01)), size = 13) +
    scale_fill_distiller(type = "div", palette = "RdYlGn", direction = 1,
                         values = valueSeq) +
    labs(title = "Plot F : Confusion Matrix for the kNN Model") +
    theme_bw(base_size = 55) +
    theme(axis.text.x = element_text(angle = 20), legend.position = "none",
          plot.title = element_text(size = 70))

tableF <- tableGrob(data.frame(Percent = round(confMatkNN$overall[1:4]*100, 2)),
                    theme = ttheme_default(base_size = 50))

plotG <- ggplot(combRFtest, aes(Reference, Prediction, fill = Freq)) +
    geom_tile(alpha = 0.7) +
    geom_text(aes(label = percent(Freq, accuracy = 0.01)), size = 13) +
    scale_fill_distiller(type = "div", palette = "RdYlGn", direction = 1,
                         values = valueSeq) +
    labs(title = "Plot G : Confusion Matrix for the combined RF model") +
    theme_bw(base_size = 55) +
    theme(axis.text.x = element_text(angle = 20), legend.position = "none",
          plot.title = element_text(size = 70))

tableG <- tableGrob(data.frame(Percent = round(confMatCombRF$overall[1:4]*100, 2)),
                    theme = ttheme_default(base_size = 50))

validationPlots <- plot_grid(plotE, tableE, NULL, NULL, plotF, tableF,
                                  NULL, NULL, plotG, tableG, ncol = 2,
                                  rel_widths = c(2,1),
                                  rel_heights = c(1,0.05,1,0.05,1)) +
    labs(caption = "Reference: Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H.\n
         Qualitative Activity Recognition of Weight Lifting Exercises\n
         Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13)\n
         Stuttgart, Germany: ACM SIGCHI, 2013") +
    theme(plot.caption = element_text(face = "italic", size = 30, hjust = 1))

print(validationPlots)
```

![](C:/Users/akash/Documents/datasciencecoursera/Weightlifting_Manner_Prediction/output/weighliftingMistakePrediction_files/figure-html/validation-1.png)<!-- -->
  
The combined random forests model with stacking was able to achieve a higher accuracy with higher bounds when compared to the other 2 models
  
<style>
div.blue { background-color:#e6f0ff; border-radius: 5px; padding: 20px;}
</style>
<div class = "blue">

# **CONCLUSION**
  
### **Thus the combined random forests model with bootstrapping controlling for out-of-bag accuracy, which stacked a gradient boost model with 5-fold repeated cross validation and a k-nearest neighbors model with 10-fold cross validation will be used to estimate the manner of weightlifting in the datToPredict data set**

</div>

  
# **MODEL APPLICATION ON THE TO-PREDICT DATA SET**
  

```r
if(system.file(package = "gt") == "") install.packages("gt")
suppressPackageStartupMessages(library(gt))

# Predictions on the datToPredictTransformed set
predictionsGBM <- predict(modelGBM, datToPredictTransformed)
predictionskNN <- predict(modelkNN, datToPredictTransformed)
toPredictDF <- data.frame(predictionsGBM, predictionskNN)
toPredictCombRF <- predict(combModelRF, toPredictDF)

# Getting the non-transformed values of the predictions which match the levels
# of the classe variable in the original data set
codeManner <- data.frame(outcome = outcomes, classe = classeLabels)
nonTransformedIndex <- match(toPredictCombRF, codeManner$outcome)
nonTransformed <- codeManner$classe[nonTransformedIndex]

# Displaying the result as a table
results <- data.frame(problem_id = datToPredictTransformed$problem_id,
                      nonTransformed = nonTransformed,
                      predictions = toPredictCombRF)
# Cleaning it up for display
results <- gt(results) %>%
    tab_header(title = md("**Predictions on the to-predict data set**")) %>%
    tab_spanner(label = md("**Mistake Predictions**"),
                columns = c(nonTransformed, predictions)) %>%
    cols_label(problem_id = md("**Problem ID**"),
               nonTransformed = md("**Original Labels**"),
               predictions = md("**Descriptive Labels**")) %>%
    cols_align(align = "center", columns = everything()) %>%
    tab_style(style = cell_borders(sides = "all", style = "solid"),
        locations = list(cells_body(columns = everything(), row = everything()),
                      cells_column_spanners(spanners = everything()),
                      cells_column_labels(columns = everything())))
# Printing the table
results
```

```{=html}
<div id="tpqwkzombn" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>#tpqwkzombn table {
  font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

#tpqwkzombn thead, #tpqwkzombn tbody, #tpqwkzombn tfoot, #tpqwkzombn tr, #tpqwkzombn td, #tpqwkzombn th {
  border-style: none;
}

#tpqwkzombn p {
  margin: 0;
  padding: 0;
}

#tpqwkzombn .gt_table {
  display: table;
  border-collapse: collapse;
  line-height: normal;
  margin-left: auto;
  margin-right: auto;
  color: #333333;
  font-size: 16px;
  font-weight: normal;
  font-style: normal;
  background-color: #FFFFFF;
  width: auto;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #A8A8A8;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #A8A8A8;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
}

#tpqwkzombn .gt_caption {
  padding-top: 4px;
  padding-bottom: 4px;
}

#tpqwkzombn .gt_title {
  color: #333333;
  font-size: 125%;
  font-weight: initial;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-color: #FFFFFF;
  border-bottom-width: 0;
}

#tpqwkzombn .gt_subtitle {
  color: #333333;
  font-size: 85%;
  font-weight: initial;
  padding-top: 3px;
  padding-bottom: 5px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-color: #FFFFFF;
  border-top-width: 0;
}

#tpqwkzombn .gt_heading {
  background-color: #FFFFFF;
  text-align: center;
  border-bottom-color: #FFFFFF;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#tpqwkzombn .gt_bottom_border {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#tpqwkzombn .gt_col_headings {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#tpqwkzombn .gt_col_heading {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 6px;
  padding-left: 5px;
  padding-right: 5px;
  overflow-x: hidden;
}

#tpqwkzombn .gt_column_spanner_outer {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  padding-top: 0;
  padding-bottom: 0;
  padding-left: 4px;
  padding-right: 4px;
}

#tpqwkzombn .gt_column_spanner_outer:first-child {
  padding-left: 0;
}

#tpqwkzombn .gt_column_spanner_outer:last-child {
  padding-right: 0;
}

#tpqwkzombn .gt_column_spanner {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 5px;
  overflow-x: hidden;
  display: inline-block;
  width: 100%;
}

#tpqwkzombn .gt_spanner_row {
  border-bottom-style: hidden;
}

#tpqwkzombn .gt_group_heading {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  text-align: left;
}

#tpqwkzombn .gt_empty_group_heading {
  padding: 0.5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: middle;
}

#tpqwkzombn .gt_from_md > :first-child {
  margin-top: 0;
}

#tpqwkzombn .gt_from_md > :last-child {
  margin-bottom: 0;
}

#tpqwkzombn .gt_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  margin: 10px;
  border-top-style: solid;
  border-top-width: 1px;
  border-top-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  overflow-x: hidden;
}

#tpqwkzombn .gt_stub {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
}

#tpqwkzombn .gt_stub_row_group {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
  vertical-align: top;
}

#tpqwkzombn .gt_row_group_first td {
  border-top-width: 2px;
}

#tpqwkzombn .gt_row_group_first th {
  border-top-width: 2px;
}

#tpqwkzombn .gt_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#tpqwkzombn .gt_first_summary_row {
  border-top-style: solid;
  border-top-color: #D3D3D3;
}

#tpqwkzombn .gt_first_summary_row.thick {
  border-top-width: 2px;
}

#tpqwkzombn .gt_last_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#tpqwkzombn .gt_grand_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#tpqwkzombn .gt_first_grand_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-style: double;
  border-top-width: 6px;
  border-top-color: #D3D3D3;
}

#tpqwkzombn .gt_last_grand_summary_row_top {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: double;
  border-bottom-width: 6px;
  border-bottom-color: #D3D3D3;
}

#tpqwkzombn .gt_striped {
  background-color: rgba(128, 128, 128, 0.05);
}

#tpqwkzombn .gt_table_body {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#tpqwkzombn .gt_footnotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#tpqwkzombn .gt_footnote {
  margin: 0px;
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}

#tpqwkzombn .gt_sourcenotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#tpqwkzombn .gt_sourcenote {
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}

#tpqwkzombn .gt_left {
  text-align: left;
}

#tpqwkzombn .gt_center {
  text-align: center;
}

#tpqwkzombn .gt_right {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

#tpqwkzombn .gt_font_normal {
  font-weight: normal;
}

#tpqwkzombn .gt_font_bold {
  font-weight: bold;
}

#tpqwkzombn .gt_font_italic {
  font-style: italic;
}

#tpqwkzombn .gt_super {
  font-size: 65%;
}

#tpqwkzombn .gt_footnote_marks {
  font-size: 75%;
  vertical-align: 0.4em;
  position: initial;
}

#tpqwkzombn .gt_asterisk {
  font-size: 100%;
  vertical-align: 0;
}

#tpqwkzombn .gt_indent_1 {
  text-indent: 5px;
}

#tpqwkzombn .gt_indent_2 {
  text-indent: 10px;
}

#tpqwkzombn .gt_indent_3 {
  text-indent: 15px;
}

#tpqwkzombn .gt_indent_4 {
  text-indent: 20px;
}

#tpqwkzombn .gt_indent_5 {
  text-indent: 25px;
}
</style>
<table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
  <thead>
    <tr class="gt_heading">
      <td colspan="3" class="gt_heading gt_title gt_font_normal gt_bottom_border" style><strong>Predictions on the to-predict data set</strong></td>
    </tr>
    
    <tr class="gt_col_headings gt_spanner_row">
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="2" colspan="1" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;" scope="col" id="&lt;strong&gt;Problem ID&lt;/strong&gt;"><strong>Problem ID</strong></th>
      <th class="gt_center gt_columns_top_border gt_column_spanner_outer" rowspan="1" colspan="2" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;" scope="colgroup" id="&lt;strong&gt;Mistake Predictions&lt;/strong&gt;">
        <span class="gt_column_spanner"><strong>Mistake Predictions</strong></span>
      </th>
    </tr>
    <tr class="gt_col_headings">
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;" scope="col" id="&lt;strong&gt;Original Labels&lt;/strong&gt;"><strong>Original Labels</strong></th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;" scope="col" id="&lt;strong&gt;Descriptive Labels&lt;/strong&gt;"><strong>Descriptive Labels</strong></th>
    </tr>
  </thead>
  <tbody class="gt_table_body">
    <tr><td headers="problem_id" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">1</td>
<td headers="nonTransformed" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">B</td>
<td headers="predictions" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">throwingElbowsFront</td></tr>
    <tr><td headers="problem_id" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">2</td>
<td headers="nonTransformed" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">A</td>
<td headers="predictions" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">noMistake</td></tr>
    <tr><td headers="problem_id" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">3</td>
<td headers="nonTransformed" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">B</td>
<td headers="predictions" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">throwingElbowsFront</td></tr>
    <tr><td headers="problem_id" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">4</td>
<td headers="nonTransformed" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">A</td>
<td headers="predictions" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">noMistake</td></tr>
    <tr><td headers="problem_id" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">5</td>
<td headers="nonTransformed" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">A</td>
<td headers="predictions" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">noMistake</td></tr>
    <tr><td headers="problem_id" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">6</td>
<td headers="nonTransformed" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">E</td>
<td headers="predictions" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">throwingHipsFront</td></tr>
    <tr><td headers="problem_id" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">7</td>
<td headers="nonTransformed" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">D</td>
<td headers="predictions" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">loweringHalfway</td></tr>
    <tr><td headers="problem_id" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">8</td>
<td headers="nonTransformed" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">B</td>
<td headers="predictions" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">throwingElbowsFront</td></tr>
    <tr><td headers="problem_id" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">9</td>
<td headers="nonTransformed" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">A</td>
<td headers="predictions" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">noMistake</td></tr>
    <tr><td headers="problem_id" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">10</td>
<td headers="nonTransformed" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">A</td>
<td headers="predictions" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">noMistake</td></tr>
    <tr><td headers="problem_id" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">11</td>
<td headers="nonTransformed" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">B</td>
<td headers="predictions" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">throwingElbowsFront</td></tr>
    <tr><td headers="problem_id" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">12</td>
<td headers="nonTransformed" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">C</td>
<td headers="predictions" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">liftingHalfway</td></tr>
    <tr><td headers="problem_id" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">13</td>
<td headers="nonTransformed" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">B</td>
<td headers="predictions" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">throwingElbowsFront</td></tr>
    <tr><td headers="problem_id" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">14</td>
<td headers="nonTransformed" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">A</td>
<td headers="predictions" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">noMistake</td></tr>
    <tr><td headers="problem_id" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">15</td>
<td headers="nonTransformed" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">E</td>
<td headers="predictions" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">throwingHipsFront</td></tr>
    <tr><td headers="problem_id" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">16</td>
<td headers="nonTransformed" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">E</td>
<td headers="predictions" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">throwingHipsFront</td></tr>
    <tr><td headers="problem_id" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">17</td>
<td headers="nonTransformed" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">A</td>
<td headers="predictions" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">noMistake</td></tr>
    <tr><td headers="problem_id" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">18</td>
<td headers="nonTransformed" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">B</td>
<td headers="predictions" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">throwingElbowsFront</td></tr>
    <tr><td headers="problem_id" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">19</td>
<td headers="nonTransformed" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">B</td>
<td headers="predictions" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">throwingElbowsFront</td></tr>
    <tr><td headers="problem_id" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">20</td>
<td headers="nonTransformed" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">B</td>
<td headers="predictions" class="gt_row gt_center" style="border-left-width: 1px; border-left-style: solid; border-left-color: #000000; border-right-width: 1px; border-right-style: solid; border-right-color: #000000; border-top-width: 1px; border-top-style: solid; border-top-color: #000000; border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">throwingElbowsFront</td></tr>
  </tbody>
  
  
</table>
</div>
```
  
# **APPENDIX**
  
## **R markdown details**
  
Written in **Rmarkdown file in R version 4.3.1 (2023-06-16 ucrt) using RStudio IDE**  
**Packages** used for this report,  
  
* **caret** : *Version 6.0.94*
* **ggplot2** : *Version 3.4.2*
* **lattice** : *Version 0.21.8*
* **dplyr** : *Version 1.1.2*
* **GGally** : *Version 2.1.2*
* **gbm** : *Version 2.1.8.1*  
* **kknn** : *Version 1.3.1*  
* **scales** : *Version 1.2.1*  
* **gridExtra** : *Version 2.3*  
* **cowplot** : *Version 1.1.1*  
* **randomForest** : *Version 4.7.1.1*  
* **gt** : *Version 0.9.0*  
  
**Creation Date of Rmarkdown file :** 2023-07-19 23:52:45.269724  
**Last Modified Date of Rmarkdown file :** 2023-07-21 22:55:56.162704
