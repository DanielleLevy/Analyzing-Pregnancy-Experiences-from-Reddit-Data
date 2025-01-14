# Step 1: Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

# Step 2: Load the dataset
# Replace with your actual file paths
df = pd.read_csv('llm_ready_dataset_labeled_with_sentiment.csv')

# Step 3: Inspect the dataset
print("Dataset preview:")
print(df.head())
print("Column names in the dataset:")
print(df.columns)

# Step 4: Data cleaning and preprocessing
# Dropping rows with missing values only in relevant columns
# Step 4: Data cleaning and preprocessing
# Update relevant columns to match dataset
relevant_columns = ['text', 'Pregnancy Week (Model)', 'neg', 'neu', 'pos', 'compound', 'Average Stress and Emotional Overload']
df_cleaned = df.dropna(subset=relevant_columns)

# Selecting relevant columns for features and target
# Using text, pregnancy week, and sentiment scores as features, and Average Stress as target
features = ['text', 'Pregnancy Week (Model)', 'neg', 'neu', 'pos', 'compound']
target = 'Average Stress and Emotional Overload'

# Ensuring all necessary columns exist
assert all(col in df_cleaned.columns for col in features + [target]), "Missing required columns in the dataset"

# Additional feature engineering
# Text length feature
df_cleaned['text_length'] = df_cleaned['text'].apply(len)
# Punctuation count (e.g., question marks)
df_cleaned['question_marks'] = df_cleaned['text'].str.count('\?')
# Ratio of negative to positive sentiment
df_cleaned['neg_pos_ratio'] = df_cleaned['neg'] / (df_cleaned['pos'] + 1e-5)

# Adding new features to the feature list
features.extend(['text_length', 'question_marks', 'neg_pos_ratio'])

# Splitting data into features (X) and target (y)
X = df_cleaned[features]
y = df_cleaned[target]

# Step 5: Text feature engineering (TF-IDF)
# Extracting TF-IDF features from text column
tfidf_vectorizer = TfidfVectorizer(max_features=100)
tfidf_matrix = tfidf_vectorizer.fit_transform(X['text']).toarray()

# Adding TF-IDF features to the dataset
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf_feature_names)
X = pd.concat([X.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
X = X.drop(columns=['text'])  # Drop the original text column

# Step 6: Recursive Feature Elimination (RFE) for feature selection
rfe_model = LinearRegression()
rfe = RFE(estimator=rfe_model, n_features_to_select=10)
rfe.fit(X, y)

# Get the selected features
selected_features = X.columns[rfe.support_]
print("\nSelected Features via RFE:")
print(selected_features)

# Step 7: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=42)

# Step 8: Model building and evaluation
# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Random Forest Regressor Model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Step 9: Evaluate models
linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_r2 = r2_score(y_test, y_pred_linear)

rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)

print("Model Evaluation Results:")
print(f"Linear Regression - MSE: {linear_mse:.2f}, R^2: {linear_r2:.2f}")
print(f"Random Forest - MSE: {rf_mse:.2f}, R^2: {rf_r2:.2f}")

# Step 10: Feature importance from Random Forest
feature_importances = pd.DataFrame({
    'Feature': selected_features,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances from Random Forest:")
print(feature_importances)

# Step 11: Visualization
# Plotting feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances from Random Forest')
plt.gca().invert_yaxis()
plt.savefig('feature_importances_random_forest.png')  # Save the plot
plt.show()

# Plotting predictions vs actual values for Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_linear, alpha=0.7, label='Linear Regression Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression: Predicted vs Actual')
plt.legend()
plt.savefig('linear_regression_predictions.png')  # Save the plot
plt.show()

# Plotting predictions vs actual values for Random Forest
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.7, label='Random Forest Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest: Predicted vs Actual')
plt.legend()
plt.savefig('random_forest_predictions.png')  # Save the plot
plt.show()
