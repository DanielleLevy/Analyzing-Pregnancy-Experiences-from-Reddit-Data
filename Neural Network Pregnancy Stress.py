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

# Adjusting SHAP value alignment and debug message
try:
    # Recalculate SHAP values to ensure alignment
    X_test_aligned = pd.DataFrame(X_test_scaled[:10], columns=X.columns[:X_test_scaled.shape[1]])
    shap_values = explainer.shap_values(X_test_scaled[:10])

    if X_test_aligned.shape[1] != len(shap_values[0][0]):
        raise ValueError("Feature and SHAP matrix dimensions are misaligned. Check the input alignment or features.")

    # Summary Plot
    shap.summary_plot(shap_values, X_test_aligned, feature_names=X.columns[:X_test_scaled.shape[1]], show=False)
    plt.savefig('shap_summary_plot_fixed.png')
    plt.close()

    # Force Plot for the first prediction
    force_plot = shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_test_aligned.iloc[0])
    shap.save_html("shap_force_plot_fixed.html", force_plot)
except ValueError as ve:
    print(f"SHAP alignment error: {ve}")

# Step 8: Visualization
# Loss Curve
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('nn_loss_curve.png')
plt.close()

# Predictions vs Actuals
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_nn, alpha=0.7, label='Neural Network Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Neural Network: Predicted vs Actual')
plt.legend()
plt.savefig('nn_predictions_vs_actuals.png')
plt.close()
