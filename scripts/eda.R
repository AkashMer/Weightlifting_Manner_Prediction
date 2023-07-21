# Loading in the data sets
training <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
                     header = TRUE, na.strings = c("","NA"))
test <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                 header = TRUE, na.strings = c("","NA"))

# Exploring the data sets
dim(training)
dim(test)
head(training$classe)
unique(training$classe)
unique(training$user_name)

# Checking if any columns are completely empty
emptyCols <- sapply(1:ncol(training), function(i) {
    mean(is.na(training[,i])) == 1
})
which(emptyCols == TRUE)

# Checking the date and time columns
unique(training$raw_timestamp_part_1)
class(training$raw_timestamp_part_1)
unique(training$raw_timestamp_part_2)

# Look at the names of variables
names(training)

# Replacing all div/0 information with 0
training$kurtosis_picth_belt[24]
sub("#DIV/0!", "0", training$kurtosis_picth_belt)
trainingDat <- matrix(NA, nrow = nrow(training), ncol = ncol(training))
for(i in 1:ncol(training)) trainingDat[,i] <- sub("#DIV/0!", 0, training[,i])
dim(trainingDat)
dim(training)

names(trainingDat)
colnames(trainingDat) <- names(training)

# Rearranging columns and converting the relevant columns to numeric
library(dplyr)
trainingDat <- as.data.frame(trainingDat)
trainingDat <- trainingDat %>%
    mutate(across(roll_belt:magnet_forearm_z, as.double)) %>%
    mutate(classe = factor(classe), new_window = factor(new_window, levels = c("no", "yes")))

# Finding the columns with calculated features
calculatedIndex <- grep("kurtosis|skewness|max|min|amplitude|avg|stddev|var",
                        names(trainingDat))
calculatedNames <- grep("kurtosis|skewness|max|min|amplitude|avg|stddev|var|gyros|accel|magnet",
                        names(trainingDat), value = TRUE)
calculatedNames <- c(calculatedNames, "classe")

# Subset out these columns for now
trainingDat <- trainingDat[,-calculatedIndex]
names(trainingDat)
mean(complete.cases(trainingDat))

# Trying a random forest model
library(caret)
modelRF <- train(classe ~ ., data = trainingDat[,-c(1,2,3,4,5,6,7)], method = "rf")
# Dont run this, takes too long

# Another approach
# Only use the data from the calculated features
trainingDat <- trainingDat %>%
    filter(new_window == "yes") %>%
    select(all_of(calculatedNames))
names(trainingDat)
mean(complete.cases(trainingDat))

summary(trainingDat)

# Removing any columns with only zeros
columnSums <- apply(trainingDat[,-141], 2, sum)
index0 <- which(columnSums == 0)

# looking at the names of these columns and Removing these columns
names(trainingDat[,index0])
trainingDat <- trainingDat[,-index0]

# Trying to fit a model
modelRF <- train(classe ~ ., data = trainingDat, method = "rf")
modelRF$finalModel

# Trying to fit a lda model
modelLDA <- train(classe ~ ., data = trainingDat, method = "lda")
modelLDA

# Trying to fit an rpart model
modelRpart <- train(classe ~ ., data = trainingDat, method = "rpart")
library(rattle)
fancyRpartPlot(modelRpart$finalModel)
modelRpart

# Seems we will have to stick to non-calculated features
library(impute)
trainingDat <- training %>%
    mutate(across(roll_belt:magnet_forearm_z, as.double)) %>%
    mutate(classe = factor(classe), new_window = factor(new_window, levels = c("no", "yes")))

# Subset out these columns for now
trainingDat <- trainingDat[,-calculatedIndex]
names(trainingDat)
mean(complete.cases(trainingDat))

summary(trainingDat[,-c(1,2,3,4,5,6,7)])

nsv <- nearZeroVar(trainingDat, saveMetrics = TRUE)
nsv

featurePlot(x = trainingDat[,-c(1,2,3,4,5,6,7,60)],
            y = trainingDat$classe)

# ------------------------------------------------------------------------------
# Starting from the beginning with a slightly different plan

training <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
                     header = TRUE, na.strings = c("","NA"))
finalTest <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                 header = TRUE, na.strings = c("","NA"))

# Partitioning the training set into, 3 sets, training(60%), validation(20%),
# and test(20%)
library(caret)
# Before just blindly splitting the data by the classe outcome, exploring the
# possibility that our data could be affected by the different time windows
windowYes <- which(training$new_window == "yes")
# Checking if the yes are equidistant
distance <- sapply(2:length(windowYes), function(i) {
    windowYes[i+1] - windowYes[i]
})
unique(distance)
as.Date(training$raw_timestamp_part_1[1:10])
as.POSIXct(training$raw_timestamp_part_1[1:40],tz = "UTC")

# Proceeding with data partition just on the outcome variable and then we will
# check the window column
set.seed(54546)
trainingIndex <- createDataPartition(training$classe, p = 0.6, list = FALSE)
train <- training[trainingIndex,]
testvalidation <- training[-trainingIndex,]
# Splitting this further to test and validation sets
testIndex <- createDataPartition(testvalidation$classe, p = 0.5, list = FALSE)
test <- testvalidation[testIndex,]
validation <- testvalidation[-testIndex,]

# Let's look at the train set
# Too many NAs
# These NAs do have a pattern, they are missing at random depending on when
# there a window slide, which is depicted by new_window = yes
# So, some of the calculated features of the data set are only calculated
# once per window slide
# Let's look at the percentage of these NAs in each column
percentNA <- sapply(1:ncol(train), function(i) mean(is.na(train[,i])))
percentNA[which(percentNA != 0)]
unique(percentNA[which(percentNA != 0)])
NAindex <- which(percentNA != 0)
names(train[,NAindex])
# So all columns with missing values have the exact same percentage of missing values
# suggesting these values as MAR instead of MCAR or MNAR
# And with such a high percentage of missing values, it would not be prudent
# to try to impute them in some way since this would reduce the variability
# of our model and increase the bias towards the selected imputation method
# for eg. imputing with values for that corresponding window would cause these
# variables to have very low variability, also, all of these are calculate features
# so we would not be losing a lot of information by ignoring these variables
# since they were derived from the measured variables which have 0 missing values
# --Limitations and Assumption--

# Applying the transformation to all the sets
train <- train[,-NAindex]
test <- test[,-NAindex]
validation <- validation[,-NAindex]
finalTestTransformed <- finalTest[,-NAindex]
mean(complete.cases(train))

# Now we have 60 variables and no missing data in our train set
# First 7 variables include, identifiers and some window sliding information
# X isjust a row identifier, user_name is the name of the individual
# timestamp columns seem like the date and milliseconds when the measured values
# were recorded, new_window information on window slide event, num_window
# seems like a window identifier
# So, X, num_window can definitely be ignored
# Removal of new_window will depend on a table
table(train$new_window, train$classe)
# Describe about the problem and explain why we can remove this
# and in turn the timestamps
# This will be one of the limitations of our models
# We will be keeping names since there might be some inter-person
# variability affecting the prediction, and if controlled for this
# our model would account for any individual

# Removing these columns from our data sets
train <- train[,-(1:7)]
test <- test[,-(1:7)]
validation <- validation[,-(1:7)]
finalTestTransformed <- finalTestTransformed[,-(1:7)]

# Class of each column in our data set is character, an effect of how read.csv
# works
# So we will be performing the following transformations
library(dplyr)
outcomes <- c("noMistake", "therowingElbowsFront", "liftingHalfway", "loweringHalfway",
              "throwingHipsFront")
train <- train %>%
    mutate(outcome = factor(classe, labels = outcomes)) %>%
    select(outcome, everything(), -classe) %>%
    mutate(across(2:53, as.numeric))
test <- test %>%
    mutate(outcome = factor(classe, labels = outcomes)) %>%
    select(outcome, everything(), -classe) %>%
    mutate(across(2:53, as.numeric))
validation <- validation %>%
    mutate(outcome = factor(classe, labels = outcomes)) %>%
    select(outcome, everything(), -classe) %>%
    mutate(across(2:53, as.numeric))
finalTestTransformed <- finalTestTransformed %>%
    select(problem_id, everything()) %>%
    mutate(across(2:53, as.numeric))

# Now let's look at the dimensions of the train data set
dim(train)

# Correlation matrix of our data set
library(GGally)
ggcorr(train)
trainNum <- train %>%
    mutate(outcome = as.numeric(outcome))
ggcorr(trainNum)
# we should pre-process with pca to remove these highly correlated variables
# we'll do this if our models don't have high accuracy

# Model with random forests
library(randomForest)
set.seed(87852)
modelRF <- train(outcome ~ ., method = "rf", data = train,
                 trControl = trainControl("oob", number = 10))
modelRF

# Model with gbm
set.seed(87852)
modelGBM <- train(outcome ~ ., method = "gbm", data = train,
                  trControl = trainControl(number = 10))
modelGBM

# Model with knn
install.packages("kknn")
library(kknn)
set.seed(87852)
modelKNN <- train(outcome ~ ., method = "kknn", data = train,
                  trControl = trainControl("cv", number = 10))
modelKNN

# Model with naive bayes classification
install.packages("klaR")
library(klaR)
set.seed(87852)
modelNBayes <- train(outcome ~ ., method = "nb", data = train,
                     trControl = trainControl("cv", number = 10))
# Not a good model

# Our models are
modelRF
modelGBM
modelKNN

# Let's predict on our test sets
predictionsRF <- predict(modelRF, test)
predictionsGBM <- predict(modelGBM, test)
predictionsKNN <- predict(modelKNN, test)

# Let's plot these
predDF <- data.frame(predictionsRF, predictionsGBM, predictionsKNN, outcome = test$outcome)
confusionMatrix(predDF$predictionsRF, test$outcome)
confusionMatrix(predictionsGBM, test$outcome)
confusionMatrix(predictionsKNN, test$outcome)

# Thus let's come up with a model which combines all 3
library(caret)
set.seed(87852)
combinedModel <- train(outcome ~ ., method = "nnet", data = predDF)
combinedModel
predictionsCombined <- predict(combinedModel, predDF)

# Let's check how good is this model
confusionMatrix(predictionsCombined, test$outcome)

# Let's check the accuracy on the validation set
predictionsRF <- predict(modelRF, validation)
predictionsGBM <- predict(modelGBM, validation)
predictionsKNN <- predict(modelKNN, validation)
validationPredDF <- data.frame(predictionsRF, predictionsGBM, predictionsKNN)
validationPredictionsCombined <- predict(combinedModel, validationPredDF)

# And drumroll...., accuracy is
confusionMatrix(validationPredDF$predictionsRF, validation$outcome)
confusionMatrix(validationPredDF$predictionsGBM, test$outcome)
confusionMatrix(validationPredDF$predictionsKNN, test$outcome)
confusionMatrix(validationPredictionsCombined, validation$outcome)

# Let's predict using these, the actual test set
dim(finalTest)
dim(finalTestTransformed)
predictionsRF <- predict(modelRF, finalTestTransformed[,-1])
predictionsGBM <- predict(modelGBM, finalTestTransformed[,-1])
predictionsKNN <- predict(modelKNN, finalTestTransformed[,-1])
finalPredDF <- data.frame(predictionsRF,
                          predictionsGBM,
                          predictionsKNN)
finalPredictionsCombined <- predict(combinedModel, finalPredDF)

# dun dun dun, dunnnnn.....
finalPredictionsCombined
identical(finalPredDF$predictionsRF, finalPredictionsCombined)

# Final answer
data.frame(finalTestTransformed$problem_id, randomForest = finalPredDF$predictionsRF,
           combined = finalPredictionsCombined)
