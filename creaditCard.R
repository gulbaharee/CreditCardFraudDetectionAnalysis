library(dplyr)
library(tidyr)
library(randomForest)
library(caret)
library(e1071)


initial_data <- read.csv("/Users/gulbahar/Desktop/courses/data mining and ml/project/creditcard_2023.csv",sep = ";")

# Assuming 'df' is your data frame
df <- initial_data[, !names(initial_data) %in% c("id")]

# Randomly sample 20,000 rows
sample_size <- 20000
df_subset <- initial_data[sample(nrow(initial_data), sample_size), ]

str(df_subset)

missing_counts <- colSums(is.na(df_subset))
missing_counts

summary(df_subset)

# Remove commas and periods, and convert to numeric
df_subset[] <- lapply(df_subset, function(x) as.numeric(gsub("[,.]", "", x)))

# Check the structure of your data
str(df_subset)


table(initial_data$Class)
# Assuming 'df' is your data frame
class_counts <- table(initial_data$Class)

barplot(class_counts, col = "skyblue", main = "Class Distribution", xlab = "Class", ylab = "Count")


# Normalize specific numeric columns
numeric_cols <- c("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                       "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
                       "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28")

correlation_matrix <- cor(df_subset[, numeric_cols])

# Create a heatmap
heatmap_plot <- ggplot(data = as.data.frame(as.table(correlation_matrix)), 
                       aes(Var1, Var2, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(Freq, 2)), vjust = 1) +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_minimal() +
  labs(title = "Correlation Heatmap of Numeric Columns")

# Print the heatmap
print(heatmap_plot)

# Normalize selected columns
df_subset[, numeric_cols] <- scale(df_subset[, numeric_cols])

# Check the structure of your data
str(df_subset)

#outliers

replace_outliers_with_mean <- function(x, threshold = 3) {
  z_scores <- abs(scale(x))
  outliers <- which(z_scores > threshold, arr.ind = TRUE)
  x[outliers] <- mean(x, na.rm = TRUE)
  return(x)
}

# Replace outliers with the mean for each selected column
for (col in numeric_cols) {
  df_subset[[col]] <- replace_outliers_with_mean(df_subset[[col]])
}


# Convert 'Class' to a factor and make levels valid R variable names
df_subset$Class <- as.factor(make.names(as.character(df_subset$Class)))

# Set the seed for reproducibility
set.seed(123)


#logistic regression
ctrl <- trainControl(method = "cv", number = 5, summaryFunction = twoClassSummary, classProbs = TRUE)

# Train the logistic regression model
logreg_model <- train(Class ~ ., data = df_subset, method = "glm", family = "binomial", trControl = ctrl)

# Print the model
print(logreg_model)

# Make predictions
logreg_predictions <- predict(logreg_model, df_subset)

# Evaluate the model
conf_matrix_logreg <- confusionMatrix(logreg_predictions, df_subset$Class)
print(conf_matrix_logreg)

#SVM

ctrl <- trainControl(method = "cv", number = 5, summaryFunction = twoClassSummary, classProbs = TRUE)

# Train the SVM model
svm_model <- train(Class ~ ., data = df_subset, method = "svmRadial", trControl = ctrl)

# Print the model
print(svm_model)

# Make predictions
svm_predictions <- predict(svm_model, df_subset)

# Evaluate the model
conf_matrix_svm <- confusionMatrix(svm_predictions, df_subset$Class)
print(conf_matrix_svm)









