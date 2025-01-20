import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib

# Step 1: Load original and new data
original_data_path = '../llm_ready_dataset_labeled_with_sentiment.csv'
pre_file_path = '../pregnancy_birth_outcomes_pre_blm.csv'
post_file_path = '../pregnancy_birth_outcomes_post_blm.csv'

original_data = pd.read_csv(original_data_path)
pre_data = pd.read_csv(pre_file_path)
post_data = pd.read_csv(post_file_path)

# Combine new datasets
new_data = pd.concat([pre_data, post_data])

# Step 2: Filter out labeled data
labeled_data = pd.read_csv(original_data_path)
unlabeled_data = new_data[~new_data['text'].isin(labeled_data['text'])]

# Step 3: Analyze original "Pregnancy Week (Model)" distribution and simulate
original_weeks = original_data['Pregnancy Week (Model)'].dropna()
week_mean = original_weeks.mean()
week_std = original_weeks.std()

# Generate simulated "Pregnancy Week (Simulated)"
unlabeled_data['Pregnancy Week (Simulated)'] = (
    pd.Series(np.random.normal(loc=week_mean, scale=week_std, size=unlabeled_data.shape[0]))
    .clip(1, 40)
    .round()
)

# Treat "Pregnancy Week (Simulated)" as "Pregnancy Week (Model)"
unlabeled_data['Pregnancy Week (Model)'] = unlabeled_data['Pregnancy Week (Simulated)']

# Step 4: Create features for new data
sia = SentimentIntensityAnalyzer()

def calculate_sentiment_features(text):
    scores = sia.polarity_scores(text)
    return pd.Series([scores['neg'], scores['neu'], scores['pos'], scores['compound']])

unlabeled_data['text'] = unlabeled_data['text'].fillna("").astype(str)
unlabeled_data[['neg', 'neu', 'pos', 'compound']] = unlabeled_data['text'].apply(calculate_sentiment_features)

# Add engineered features
unlabeled_data['text_length'] = unlabeled_data['text'].apply(len)
unlabeled_data['question_marks'] = unlabeled_data['text'].str.count('\?')
unlabeled_data['neg_pos_ratio'] = unlabeled_data['neg'] / (unlabeled_data['pos'] + 1e-5)
unlabeled_data['week_neg_interaction'] = unlabeled_data['Pregnancy Week (Model)'] * unlabeled_data['neg']

# Step 5: Load the best model
best_model_path = '../ex4_improve_models/best_model.pkl'
best_model = joblib.load(best_model_path)

# Step 6: Predict with the best model
features = ['Pregnancy Week (Model)', 'neg', 'neu', 'pos', 'compound',
            'text_length', 'question_marks', 'neg_pos_ratio', 'week_neg_interaction']
predictions = best_model.predict(unlabeled_data[features])

unlabeled_data['Predicted Stress'] = predictions

# Step 7: Save the predictions
unlabeled_data[['text', 'Pregnancy Week (Simulated)', 'Predicted Stress'] + features].to_csv(
    'new_data_with_predictions.csv', index=False
)

print("Predictions saved to 'new_data_with_predictions.csv'.")

# Step 8: Visualizations
# 1. Distribution of Predicted Stress
plt.figure(figsize=(10, 6))
sns.histplot(unlabeled_data['Predicted Stress'], kde=True, color='blue')
plt.title('Distribution of Predicted Stress')
plt.xlabel('Predicted Stress')
plt.ylabel('Frequency')
plt.savefig('predicted_stress_distribution.png')
plt.close()

# 2. Correlation heatmap between features and Predicted Stress
correlation_matrix = unlabeled_data[features + ['Predicted Stress']].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig('feature_correlation_heatmap.png')
plt.close()

# 3. Scatter plots for key features vs Predicted Stress
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=unlabeled_data[feature], y=unlabeled_data['Predicted Stress'], alpha=0.6)
    plt.title(f'{feature} vs Predicted Stress')
    plt.xlabel(feature)
    plt.ylabel('Predicted Stress')
    plt.grid(True)
    plt.savefig(f'{feature}_vs_predicted_stress.png')
    plt.close()

print("All visualizations saved.")
