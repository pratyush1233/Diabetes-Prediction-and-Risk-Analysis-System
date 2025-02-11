# Load required libraries
library(ggplot2)
library(caret)
library(e1071)  # For Naive Bayes and SVM
library(randomForest)
library(class)  # For KNN

# Load data
data <- read.csv("A:/Repositories/Capstone/data/processed/diabetes.csv")

# Rename columns
colnames(data)[colnames(data) == "BloodPressure"] <- "BP"
colnames(data)[colnames(data) == "Glucose"] <- "Sugar"
colnames(data)[colnames(data) == "Outcome"] <- "Condition"

# Map 'Condition' column to categorical labels
data$Condition <- factor(data$Condition, levels = c(0, 1), labels = c("Normal", "Diabetic"))

# Select features and target
X <- data[c("BP", "Sugar", "BMI", "Age")]
y <- data$Condition

# Standardize features
preProc <- preProcess(X, method = c("center", "scale"))
X_scaled <- predict(preProc, X)

# Split data into training and testing sets
set.seed(42)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X_scaled[trainIndex, ]
X_test <- X_scaled[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

# Define models
models <- list(
  "Naive Bayes" = naiveBayes(X_train, y_train),
  "Decision Tree" = rpart::rpart(y_train ~ ., data = data.frame(X_train, y_train), method = "class"),
  "Random Forest" = randomForest(X_train, y_train, ntree = 100, random_state = 42),
  "SVM" = svm(y_train ~ ., data = data.frame(X_train, y_train)),
  "KNN" = knn3(X_train, y_train, k = 5)
)

# Evaluate models and store accuracies
accuracies <- sapply(names(models), function(name) {
  model <- models[[name]]
  if (name == "KNN") {
    y_pred <- predict(model, X_test)
  } else {
    y_pred <- predict(model, newdata = data.frame(X_test), type = "class")
  }
  accuracy <- mean(y_pred == y_test)
  cat(paste(name, "Accuracy:", round(accuracy, 2), "\n"))
  return(accuracy)
})

# Plot accuracies
accuracy_df <- data.frame(Model = names(accuracies), Accuracy = unlist(accuracies))
ggplot(accuracy_df, aes(x = Model, y = Accuracy)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black", width = 0.5) +
  theme_minimal() +
  labs(title = "Model Accuracies", x = "Model", y = "Accuracy")

# Save the best model
best_model_name <- names(accuracies)[which.max(accuracies)]
best_model <- models[[best_model_name]]
saveRDS(best_model, "best_health_model.rds")
cat(paste("Best model (", best_model_name, ") saved.\n"))

# Load the best model
best_model <- readRDS("best_health_model.rds")

# Predict for user input
user_input <- data.frame(BP = 140, Sugar = 110, Age = 50, BMI = 28)
user_features <- predict(preProc, user_input)
risk_zone <- predict(best_model, user_features)
cat("\nPredicted Risk Zone:", risk_zone, "\n")

# Compare with thresholds
thresholds <- list(
  Normal = c(BP = 120, Sugar = 100, HeartRate = 70),
  "Pre-Diabetic" = c(BP = 130, Sugar = 110, HeartRate = 80),
  Diabetic = c(BP = 140, Sugar = 126, HeartRate=90)
)

compare_with_thresholds <- function(user_input, thresholds) {
  for (condition in names(thresholds)) {
    limits <- thresholds[[condition]]
    within_limits <- all(sapply(names(limits), function(param) {
      if (param %in% names(user_input)) user_input[[param]] <= limits[param] else TRUE
    }))
    if (within_limits) return(condition)
  }
  return("Above All Limits")
}
feedback <- compare_with_thresholds(user_input, thresholds)
cat("Threshold-Based Condition:", feedback, "\n")

# Blood Pressure Comparison Plot
bp_data <- data.frame(BP = data$BP)
ggplot(bp_data, aes(x = BP)) +
  geom_histogram(bins = 20, fill = "skyblue", alpha = 0.7) +
  geom_vline(xintercept = user_input$BP, color = "red", linetype = "dashed", size = 1) +
  labs(title = "Blood Pressure Comparison", x = "Blood Pressure", y = "Frequency") +
  theme_minimal()
