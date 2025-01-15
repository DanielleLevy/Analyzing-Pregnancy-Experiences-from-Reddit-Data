# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import shap
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv('llm_ready_dataset_labeled_with_sentiment.csv')

# Step 2: Data cleaning and preprocessing
# Dropping rows with missing values
relevant_columns = ['text', 'Pregnancy Week (Model)', 'neg', 'neu', 'pos', 'compound', 'Average Stress and Emotional Overload']
df_cleaned = df.dropna(subset=relevant_columns)

# Feature engineering
df_cleaned['text_length'] = df_cleaned['text'].apply(len)
df_cleaned['question_marks'] = df_cleaned['text'].str.count('\?')
df_cleaned['neg_pos_ratio'] = df_cleaned['neg'] / (df_cleaned['pos'] + 1e-5)
df_cleaned['week_neg_interaction'] = df_cleaned['Pregnancy Week (Model)'] * df_cleaned['neg']

# Splitting data into features and target
features = ['Pregnancy Week (Model)', 'neg', 'neu', 'pos', 'compound', 'text_length', 'question_marks', 'neg_pos_ratio', 'week_neg_interaction']
X_base = df_cleaned[features]
y = df_cleaned['Average Stress and Emotional Overload']

# TF-IDF for text column
tfidf_vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
tfidf_matrix = tfidf_vectorizer.fit_transform(df_cleaned['text']).toarray()

# Adding TF-IDF features to X
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf_feature_names)
X = pd.concat([X_base.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Build Neural Network Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Step 5: Train the Model
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

# Step 6: Evaluate the Model
evaluation = model.evaluate(X_test_scaled, y_test, verbose=0)
y_pred_nn = model.predict(X_test_scaled)
mse_nn = mean_squared_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)

print(f"Neural Network - MSE: {mse_nn:.2f}, R^2: {r2_nn:.2f}")

# Step 7: SHAP Analysis
explainer = shap.KernelExplainer(model.predict, X_train_scaled[:100])
shap_values = explainer.shap_values(X_test_scaled[:10])

# Aligning SHAP values and features
aligned_columns = X.columns[:X_test_scaled.shape[1]]
X_test_aligned = pd.DataFrame(X_test_scaled[:10], columns=aligned_columns)

assert X_test_aligned.shape[1] == len(shap_values[0]), "Feature and SHAP matrices must have the same number of rows!"

# Summary Plot
shap.summary_plot(shap_values, X_test_aligned, feature_names=aligned_columns, show=False)
plt.savefig('Feature_Impact_Analysis_SHAP.png')
plt.close()

# Force Plot for the first prediction
force_plot = shap.force_plot(explainer.expected_value, shap_values[0][0], X_test_aligned.iloc[0, :].values, feature_names=aligned_columns)
shap.save_html("SHAP_Force_Plot_Specific_Prediction.html", force_plot)

# Step 8: Visualization
# Loss Curve
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve Across Training and Validation Phases')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Loss_Curve_Training_Validation.png')
plt.close()

# Predictions vs Actuals
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_nn, alpha=0.7, label='Neural Network Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Stress Levels')
plt.legend()
plt.savefig('Predicted_vs_Actual_Stress_Levels.png')
plt.close()
