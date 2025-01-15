# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import numpy as np

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

# Step 3: Recursive Feature Elimination (RFE)
rfe_model = LinearRegression()
rfe = RFE(estimator=rfe_model, n_features_to_select=10)
rfe.fit(X, y)
selected_features = X.columns[rfe.support_]
print("Selected Features via RFE:", selected_features)

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure feature alignment between train and test sets
X_test = X_test[selected_features]

# Step 5: Model Training and Evaluation
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'LightGBM': LGBMRegressor(random_state=42)
}

results = {}

for model_name, model in models.items():
    model.fit(X_train[selected_features], y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[model_name] = {'MSE': mse, 'R2': r2}
    print(f"{model_name} - MSE: {mse:.2f}, R^2: {r2:.2f}")

# Step 6: Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for model_name, model in models.items():
    cv_mse = []
    cv_r2 = []
    for train_index, test_index in kf.split(X):
        X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
        y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
        X_test_cv = X_test_cv[selected_features]
        model.fit(X_train_cv[selected_features], y_train_cv)
        y_pred_cv = model.predict(X_test_cv)
        cv_mse.append(mean_squared_error(y_test_cv, y_pred_cv))
        cv_r2.append(r2_score(y_test_cv, y_pred_cv))
    print(f"{model_name} Cross-Validation - Average MSE: {np.mean(cv_mse):.2f}, Average R^2: {np.mean(cv_r2):.2f}")

# Step 7: Visualizations
# Feature Importance for Random Forest
rf_model = models['Random Forest']
rf_feature_importances = rf_model.feature_importances_[:len(selected_features)]
feature_importances = pd.DataFrame({
    'Feature': selected_features,
    'Importance': rf_feature_importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances from Random Forest')
plt.gca().invert_yaxis()
plt.savefig('feature_importances_random_forest.png')
plt.show()

# Predictions vs Actuals for Linear Regression
lr_model = models['Linear Regression']
y_pred_lr = lr_model.predict(X_test)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.7, label='Linear Regression Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression: Predicted vs Actual')
plt.legend()
plt.savefig('linear_regression_predictions_vs_actuals.png')
plt.show()

# Predictions vs Actuals for LightGBM
lgbm_model = models['LightGBM']
y_pred_lgbm = lgbm_model.predict(X_test)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lgbm, alpha=0.7, label='LightGBM Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('LightGBM: Predicted vs Actual')
plt.legend()
plt.savefig('lightgbm_predictions_vs_actuals.png')
plt.show()
