install.packages("readxl")
library(readxl)
install.packages("fastDummies")
library(fastDummies)
library(dplyr)
set.seed(123)


#Reading data from the table
table <- read_excel("D:/Data Science for Business Project/cleaned_player_data.xlsx")


#Segregating the table based on position
table_attack <- table %>% filter(Position == "attack")
table_midfield <- table %>% filter(Position == "midfield")  
table_Goalkeeper <- table %>% filter(Position == "Goalkeeper") 
table_Defender <- table %>% filter(Position == "Defender") 

#Dummy columns
table_attack <- fastDummies::dummy_cols(table_attack, select_columns = c("Club", "Position","Nation","League"), remove_first_dummy = TRUE)
table_midfield <- fastDummies::dummy_cols(table_midfield, select_columns = c("Club", "Position","Nation","League"), remove_first_dummy = TRUE)
table_Goalkeeper <- fastDummies::dummy_cols(table_Goalkeeper, select_columns = c("Club", "Position","Nation","League"), remove_first_dummy = TRUE)
table_Defender <- fastDummies::dummy_cols(table_Defender, select_columns = c("Club", "Position","Nation","League"), remove_first_dummy = TRUE)

#Removing the columns for which we have dummy columns
new_table_attack <- table_attack %>%select(-c(Player, Club, Position, Nation, League, `"Squad(20/21)"`))
new_table_midfield <- table_midfield %>%select(-c(Player, Club, Position, Nation, League, `"Squad(20/21)"`))
new_table_Goalkeeper <- table_Goalkeeper %>%select(-c(Player, Club, Position, Nation, League, `"Squad(20/21)"`))
new_table_Defender <- table_Defender %>%select(-c(Player, Club, Position, Nation, League, `"Squad(20/21)"`))

#Splitting test and train data
table_indices <- sample(1:nrow(new_table_attack), size = 0.7 * nrow(new_table_attack))
table_attack <- new_table_attack[table_indices, ]
test_attack_data <- new_table_attack[-table_indices, ]

table_indices <- sample(1:nrow(new_table_midfield), size = 0.7 * nrow(new_table_midfield))
table_midfield <- new_table_midfield[table_indices, ]
test_midfield_data <- new_table_midfield[-table_indices, ]

table_indices <- sample(1:nrow(new_table_Defender), size = 0.7 * nrow(new_table_Defender))
table_defense <- new_table_Defender[table_indices, ]
test_defense_data <- new_table_Defender[-table_indices, ]

table_indices <- sample(1:nrow(new_table_Goalkeeper), size = 0.7 * nrow(new_table_Goalkeeper))
table_goalkeeper <- new_table_Goalkeeper[table_indices, ]
test_goalkeeper_data <- new_table_Goalkeeper[-table_indices, ]

#Running the model for attack
model_attack <- lm(Value ~ ., data = table_attack)
x <-predict(model_attack, newdata = test_attack_data)
mse <- mean((table_attack$Value - x)^2)
rmse <- sqrt(mse)
rmse

#Running the model for midfield
model_midfield <- lm(Value ~ ., data = table_midfield)
midfield_predict <-predict(model_midfield, newdata = test_midfield_data)
mse <- mean((table_midfield$Value - midfield_predict)^2)
rmse_midfield <- sqrt(mse)
rmse_midfield

#Running the model for defense
model_Defender <- lm(Value ~ ., data = table_defense)
defender_predict <-predict(model_Defender, newdata = test_defense_data)
mse_defender <- mean((table_defense$Value - defender_predict)^2)
rmse_defender <- sqrt(mse_defender)
rmse_defender

#Running the model for goal keeper
model_Goalkeeper <- lm(Value ~ ., data = table_goalkeeper)
goalkeeper_predict <-predict(model_Goalkeeper, newdata = test_goalkeeper_data)
mse_goalkeeper <- mean((table_goalkeeper$Value - goalkeeper_predict)^2)
rmse_goalkeeper <- sqrt(mse_goalkeeper)
rmse_goalkeeper




#Lasso Regression
install.packages("dplyr")
library(dplyr)
install.packages("glmnet")
library(glmnet)
set.seed(123)
install.packages("caTools")
library(caTools)
install.packages("caret")
library(caret)
install.packages("Metrics")
library(Metrics)


# Split data: 80% train, 20% test
train_index <- createDataPartition(new_table_attack$Value, p = 0.8, list = FALSE)
train_data <- new_table_attack[train_index, ]
test_data <- new_table_attack[-train_index, ]


unique_values <- sapply(train_data, function(x) length(unique(x)))
print(unique_values)

x_train <- as.matrix(train_data %>% select(-Value))
y_train <- train_data$Value
x_test <- as.matrix(test_data %>% select(-Value))
y_test <- test_data$Value

lasso_model <- cv.glmnet(x_train, y_train, alpha = 1)
best_lambda <- lasso_model$lambda.min
lasso_final <- glmnet(x_train, y_train, alpha = 1, lambda = best_lambda)

# Get the coefficients
lasso_coefficients <- coef(lasso_final)
print(lasso_coefficients)

lasso_df <- as.data.frame(as.matrix(lasso_coefficients))

# Add a column for coefficient names
lasso_df$Feature <- rownames(lasso_df)

# Rename the coefficient value column (usually it's "1")
colnames(lasso_df)[1] <- "Coefficient"

# Sort the coefficients in descending order
lasso_df_sorted <- lasso_df[order(-lasso_df$Coefficient), ]

predictions_attack <- predict(lasso_final, s = best_lambda, newx = x_test)

# View the sorted coefficients
print(lasso_df_sorted)
rmse_value <- rmse(y_test, predictions_attack)
rmse_value

# Split data: 80% train, 20% test For defenders-------------------------------------------------------------------
new_table_Defender <- na.omit(new_table_Defender)
train_index_defender <- createDataPartition(new_table_Defender$Value, p = 0.8, list = FALSE)
train_data_defender <- new_table_Defender[train_index_defender, ]
test_data_defender <- new_table_Defender[-train_index_defender, ]


unique_values_defender <- sapply(train_index_defender, function(x) length(unique(x)))
print(unique_values_defender)

x_train_defender <- as.matrix(train_data_defender %>% select(-Value))
y_train_defender <- train_data_defender$Value
x_test_defender <- as.matrix(test_data_defender %>% select(-Value))
y_test_defender <- test_data_defender$Value


lasso_model_defender <- cv.glmnet(x_train_defender, y_train_defender, alpha = 1)
best_lambda_defender <- lasso_model_defender$lambda.min
lasso_final_defender <- glmnet(x_train_defender, y_train_defender, alpha = 1, lambda = best_lambda_defender)

# Get the coefficients
lasso_coefficients_defender <- coef(lasso_final_defender)
print(lasso_coefficients_defender)

lasso_df_defender <- as.data.frame(as.matrix(lasso_coefficients_defender))

# Add a column for coefficient names
lasso_df_defender$Feature <- rownames(lasso_df_defender)

# Rename the coefficient value column (usually it's "1")
colnames(lasso_df_defender)[1] <- "Coefficient"

# Sort the coefficients in descending order
lasso_df_sorted_defender <- lasso_df_defender[order(-lasso_df_defender$Coefficient), ]

predictions_defender <- predict(lasso_final_defender, s = best_lambda_defender, newx = x_test_defender)

# View the sorted coefficients
print(lasso_df_sorted_defender)
rmse_defender <- rmse(y_test_defender, predictions_defender)
rmse_defender

# Split data: 80% train, 20% test For midfielder-------------------------------------------------------------------
new_table_midfield <- na.omit(new_table_midfield)
train_index_midfield <- createDataPartition(new_table_midfield$Value, p = 0.8, list = FALSE)
train_data_midfield <- new_table_midfield[train_index_midfield, ]
test_data_midfield <- new_table_midfield[-train_index_midfield, ]


unique_values_midfielder <- sapply(train_index_midfield, function(x) length(unique(x)))
print(unique_values_midfielder)

x_train_midfielder <- as.matrix(train_data_midfield %>% select(-Value))
y_train_midfielder <- train_data_midfield$Value
x_test_midfielder <- as.matrix(test_data_midfield %>% select(-Value))
y_test_midfielder <- test_data_midfield$Value


lasso_model_midfielder <- cv.glmnet(x_train_midfielder, y_train_midfielder, alpha = 1)
best_lambda_midfielder <- lasso_model_midfielder$lambda.min
lasso_final_midfielder <- glmnet(x_train_midfielder, y_train_midfielder, alpha = 1, lambda = best_lambda_midfielder)

# Get the coefficients
lasso_coefficients_midfielder <- coef(lasso_final_midfielder)
print(lasso_coefficients_midfielder)

lasso_df_midfielder <- as.data.frame(as.matrix(lasso_coefficients_midfielder))

# Add a column for coefficient names
lasso_df_midfielder$Feature <- rownames(lasso_df_midfielder)

# Rename the coefficient value column (usually it's "1")
colnames(lasso_df_midfielder)[1] <- "Coefficient"

# Sort the coefficients in descending order
lasso_df_sorted_midfielder <- lasso_df_midfielder[order(-lasso_df_midfielder$Coefficient), ]

predictions_midfielder <- predict(lasso_final_midfielder, s = best_lambda_midfielder, newx = x_train_midfielder)

# View the sorted coefficients
print(lasso_df_sorted_midfielder)
rmse_midfielder <- rmse(y_test_midfielder, predictions_midfielder)
rmse_midfielder

# Random Forest for regression for attackers

install.packages("randomForest")
library(randomForest)
set.seed(123)
colnames(new_table_attack) <- make.names(colnames(new_table_attack))

train_index_attack <- sample(1:nrow(new_table_attack), 0.8 * nrow(new_table_attack))
train_data_attack <- new_table_attack[train_index_attack, ]
test_data_attack <- new_table_attack[-train_index_attack, ]
rf_model_attack <- randomForest(Value ~ ., data = train_data_attack, importance = TRUE)
predictions_attack <- predict(rf_model_attack, newdata = test_data_attack)
actual_attack <- test_data_attack$Value
rmse_attack <- sqrt(mean((predictions_attack - actual_attack)^2))
#Finding important coefficients for attacker
importance_lasso_attacker_rf <- importance(rf_model_attack)
importance_df_attacker <- as.data.frame(importance_lasso_attacker_rf)
sorted_importance_attacker <- importance_df_attacker[order(-importance_df_attacker[, "%IncMSE"]), ]
top_10_variables_attacker <- head(sorted_importance_attacker, 10)
top_10_variables_attacker

# Random Forest for regression for midfielders

library(randomForest)
set.seed(123)
colnames(new_table_midfield) <- make.names(colnames(new_table_midfield))

train_index_midfield <- sample(1:nrow(new_table_midfield), 0.8 * nrow(new_table_midfield))
train_data_midfield <- new_table_midfield[train_index_midfield, ]
test_data_midfield <- new_table_midfield[-train_index_midfield, ]
rf_model_midfield <- randomForest(Value ~ ., data = train_data_midfield, importance = TRUE)
predictions_midfield <- predict(rf_model_midfield, newdata = test_data_midfield)
actual_midfield <- test_data_midfield$Value
rmse_midfield <- sqrt(mean((predictions_midfield - actual_midfield)^2))
#Finding important coefficients for defenders
importance_lasso_midfielder_rf <- importance(rf_model_midfield)
importance_df_midfielder <- as.data.frame(importance_lasso_midfielder_rf)
sorted_importance_midfielder <- importance_df_midfielder[order(-importance_df_midfielder[, "%IncMSE"]), ]
top_10_variables_midfielder <- head(sorted_importance_midfielder, 10)
top_10_variables_midfielder

# Random Forest for regression for defenders

library(randomForest)
set.seed(123)
colnames(new_table_Defender) <- make.names(colnames(new_table_Defender))
train_index_Defender <- sample(1:nrow(new_table_Defender), 0.8 * nrow(new_table_Defender))
train_data_Defender <- new_table_Defender[train_index_Defender, ]
test_data_Defender <- new_table_Defender[-train_index_Defender, ]
rf_model_Defender <- randomForest(Value ~ ., data = train_data_Defender, importance = TRUE)
predictions_Defender <- predict(rf_model_Defender, newdata = test_data_Defender)
actual_Defender <- test_data_Defender$Value
rmse_Defender <- sqrt(mean((predictions_Defender - actual_Defender)^2))
#Finding important coefficients for defenders
importance_lasso_defender_rf <- importance(rf_model_Defender)
importance_df_defender <- as.data.frame(importance_lasso_defender_rf)
sorted_importance_defender <- importance_df_defender[order(-importance_df_defender[, "%IncMSE"]), ]
top_10_variables_defender <- head(sorted_importance_defender, 10)
top_10_variables_defender


#Predict Manchester Uniteds actual value for defenders-----------------------------------------------------
new_table_manchester_Defender <- new_table_Defender %>% filter(Club_Manchester.United == 1)
predictions_manchester_Defender <- predict(rf_model_Defender, newdata = new_table_manchester_Defender)
sum_predictions_manchester_Defender <- sum(predictions_manchester_Defender)
sum_predictions_manchester_Defender
actual_table_manchester_Defender <- new_table_Defender %>% filter(Club_Manchester.United == 1)
sum_actual_manchester_Defender <- sum(actual_table_manchester_Defender$Value)
sum_actual_manchester_Defender

#Predict Manchester Uniteds actual value for midfielders-----------------------------------------------------
new_table_manchester_Midfielder <- new_table_midfield %>% filter(Club_Manchester.United == 1)
predictions_manchester_Midfielder <- predict(rf_model_midfield, newdata = new_table_manchester_Midfielder)
sum_predictions_manchester_Midfielder <- sum(predictions_manchester_Midfielder)
sum_predictions_manchester_Midfielder
actual_table_manchester_Midfielder <- new_table_midfield %>% filter(Club_Manchester.United == 1)
sum_actual_manchester_Midfielder <- sum(actual_table_manchester_Midfielder$Value)
sum_actual_manchester_Midfielder

#Predict Manchester Uniteds actual value for attackers-----------------------------------------------------
new_table_manchester_Attacker <- new_table_attack %>% filter(Club_Manchester.United == 1)
predictions_manchester_Attacker <- predict(rf_model_attack, newdata = new_table_manchester_Attacker)
sum_predictions_manchester_Attacker <- sum(predictions_manchester_Attacker)
sum_predictions_manchester_Attacker
actual_table_manchester_Attacker <- new_table_attack %>% filter(Club_Manchester.United == 1)
sum_actual_manchester_Attacker <- sum(actual_table_manchester_Attacker$Value)
sum_actual_manchester_Attacker

#K fold cross validation for attackers lasso--------------------------------------------------------------
# Load necessary packages
install.packages("caret")
library(caret)
library(glmnet)
install.packages("dplyr")
library(dplyr)
# Assuming new_table_attack is already prepared
set.seed(123)
# Prepare data
x <- as.matrix(new_table_attack %>% select(-Value))
y <- new_table_attack$Value
# Define control for k-fold cross-validation
control <- trainControl(method = "cv", number = 10)  # 10-fold cross-validation
# Train Lasso regression model with cross-validation
lasso_model_cv <- train(x, y,
                        method = "glmnet",
                        trControl = control,
                        tuneLength = 10)  # This specifies how many lambda values to try
# Display results
print(lasso_model_cv)
# Get best lambda value
best_lambda <- lasso_model_cv$bestTune$lambda
# Make predictions
predictions <- predict(lasso_model_cv, newdata = x)
# Calculate RMSE
rmse_value <- sqrt(mean((y - predictions)^2))
print(paste("RMSE: ", rmse_value))

#K fold cross validation for midfielders lasso
set.seed(123)
# Prepare data
x_midfield <- as.matrix(new_table_midfield %>% select(-Value))
y_midfield <- new_table_midfield$Value
# Define control for k-fold cross-validation
control <- trainControl(method = "cv", number = 10)  # 10-fold cross-validation
# Train Lasso regression model with cross-validation
lasso_model_cv_midfield <- train(x_midfield, y_midfield,
                        method = "glmnet",
                        trControl = control,
                        tuneLength = 10)  # This specifies how many lambda values to try
# Display results
print(lasso_model_cv_midfield)
# Get best lambda value
best_lambda <- lasso_model_cv_midfield$bestTune$lambda
# Make predictions
predictions_midfield <- predict(lasso_model_cv_midfield, newdata = x_midfield)
# Calculate RMSE
rmse_value_midfield <- sqrt(mean((y_midfield - predictions_midfield)^2))
print(paste("RMSE: ", rmse_value_midfield))

#K fold cross validation for defenders lasso
set.seed(123)
# Prepare data
x_defender <- as.matrix(new_table_Defender %>% select(-Value))
y_defender <- new_table_Defender$Value
# Define control for k-fold cross-validation
control <- trainControl(method = "cv", number = 10)  # 10-fold cross-validation
# Train Lasso regression model with cross-validation
lasso_model_cv_defender <- train(x_defender, y_defender,
                                 method = "glmnet",
                                 trControl = control,
                                 tuneLength = 10)  # This specifies how many lambda values to try
# Display results
print(lasso_model_cv_defender)
# Get best lambda value
best_lambda <- lasso_model_cv_defender$bestTune$lambda
# Make predictions
predictions_defender <- predict(lasso_model_cv_defender, newdata = x_defender)
# Calculate RMSE
rmse_value_defender <- sqrt(mean((y_defender - predictions_defender)^2))
print(paste("RMSE: ", rmse_value_defender))

#K fold cross validation for attacker random forest------------------------------------------------------------------
# Set up training control for k-fold cross-validation
train_control <- trainControl(method = "cv", number = 10) # 10-fold cross-validation

# Train Random Forest model with cross-validation
rf_model_attack_cf <- train(Value ~ ., data = new_table_attack, method = "rf", trControl = train_control)

# Output the model results
print(rf_model_attack)

# Make predictions on the training data to calculate RMSE
predictions_attack <- predict(rf_model_attack, newdata = new_table_attack)
actual_attack <- new_table_attack$Value
rmse_attack <- sqrt(mean((predictions_attack - actual_attack)^2))
cat("RMSE for attack model with k-fold CV:", rmse_attack, "\n")

#K fold cross validation for defender random forest---------------------------------
# Set up training control for k-fold cross-validation
train_control <- trainControl(method = "cv", number = 10) # 10-fold cross-validation

# Train Random Forest model with cross-validation
rf_model_defender_cf <- train(Value ~ ., data = new_table_Defender, method = "rf", trControl = train_control)

# Output the model results
print(rf_model_defender_cf)

# Make predictions on the training data to calculate RMSE
predictions_defender <- predict(rf_model_defender_cf, newdata = new_table_Defender)
actual_defender <- new_table_Defender$Value
rmse_defender <- sqrt(mean((predictions_defender - actual_defender)^2))
cat("RMSE for defender model with k-fold CV:", rmse_defender, "\n")

#K fold cross validation for midfielder random forest---------------------------------
# Set up training control for k-fold cross-validation
train_control <- trainControl(method = "cv", number = 10) # 10-fold cross-validation

# Train Random Forest model with cross-validation
rf_model_midfield_cf <- train(Value ~ ., data = new_table_midfield, method = "rf", trControl = train_control)

# Output the model results
print(rf_model_midfield_cf)

# Make predictions on the training data to calculate RMSE
predictions_midfield <- predict(rf_model_midfield_cf, newdata = new_table_midfield)
actual_midfielder <- new_table_midfield$Value
rmse_midfielder <- sqrt(mean((predictions_midfield - actual_midfielder)^2))
cat("RMSE for midfielder model with k-fold CV:", rmse_midfielder, "\n")
rmse_midfielder_2 <- rf_model_midfield_cf$resample$RMSE
print(paste("Random Forest OOS RMSE:", mean(rmse_midfielder_2)))

