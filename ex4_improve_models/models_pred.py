# Import necessary libraries
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import lightgbm as lgb

# Load the dataset
# Replace 'your_dataset.csv' with the actual file path
file_path = '../llm_ready_dataset_labeled_with_sentiment.csv'
df = pd.read_csv(file_path)

# Clean the dataset
relevant_columns = ['Pregnancy Week (Model)', 'neg', 'neu', 'pos', 'compound',
                    'Average Stress and Emotional Overload', 'text']
df_cleaned = df.dropna(subset=relevant_columns)

# Feature engineering
df_cleaned['text_length'] = df_cleaned['text'].apply(len)
df_cleaned['question_marks'] = df_cleaned['text'].str.count('\?')
df_cleaned['neg_pos_ratio'] = df_cleaned['neg'] / (df_cleaned['pos'] + 1e-5)
df_cleaned['week_neg_interaction'] = df_cleaned['Pregnancy Week (Model)'] * df_cleaned['neg']

# Define features and target
features = ['Pregnancy Week (Model)', 'neg', 'neu', 'pos', 'compound',
            'text_length', 'question_marks', 'neg_pos_ratio', 'week_neg_interaction']
target = 'Average Stress and Emotional Overload'

X = df_cleaned[features]
y = df_cleaned[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate and plot models
def evaluate_model(model, model_name):
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    correlation, _ = pearsonr(y_test, y_pred)

    # Plot predictions vs actual values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.title(f'{model_name} Predictions vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.savefig(f'{model_name}_predictions_vs_actual.png')
    plt.close()

    print(f"{model_name} - MSE: {mse:.2f}, R2: {r2:.2f}, Correlation: {correlation:.2f}")

    return mse, r2, correlation

# Perform Hyperparameter Tuning for Gradient Boosting
def tune_gradient_boosting():
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    gb_model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=5, scoring='r2', verbose=2)
    grid_search.fit(X_train, y_train)
    print("Best Parameters:", grid_search.best_params_)
    print("Best R2 Score:", grid_search.best_score_)
    return grid_search.best_estimator_

# Perform Hyperparameter Tuning for Random Forest
def tune_random_forest():
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    rf_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='r2', verbose=2)
    grid_search.fit(X_train, y_train)
    print("Best Parameters:", grid_search.best_params_)
    print("Best R2 Score:", grid_search.best_score_)
    return grid_search.best_estimator_

# Perform Hyperparameter Tuning for Support Vector Regressor
def tune_svr():
    param_grid = {
        'C': [0.1, 1, 10],
        'epsilon': [0.1, 0.2, 0.5],
        'kernel': ['linear', 'rbf']
    }
    svr_model = SVR()
    grid_search = GridSearchCV(estimator=svr_model, param_grid=param_grid, cv=5, scoring='r2', verbose=2)
    grid_search.fit(X_train, y_train)
    print("Best Parameters:", grid_search.best_params_)
    print("Best R2 Score:", grid_search.best_score_)
    return grid_search.best_estimator_

# Analyze Feature Importance
def plot_feature_importance(model, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Plot feature importance
        plt.figure(figsize=(16, 12))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title(f'Feature Importance ({model_name})')
        plt.savefig(f'{model_name}_feature_importance.png')
        plt.close()

# Models to evaluate
models = {
    'Random Forest': tune_random_forest(),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    'Linear Regression': LinearRegression(),
    'Gradient Boosting': tune_gradient_boosting(),
    'Elastic Net': ElasticNet(random_state=42),
    'Support Vector Regressor': tune_svr(),
    'LightGBM': lgb.LGBMRegressor(random_state=42)
}

# Store results and save the best model
results = []
best_model = None
best_model_name = ""
best_r2 = -float("inf")  # להתחיל עם ערך מינימלי כדי להשוות

for model_name, model in models.items():
    mse, r2, correlation = evaluate_model(model, model_name)
    results.append({'Model': model_name, 'MSE': mse, 'R2': r2, 'Correlation': correlation})
    plot_feature_importance(model, model_name)

    # Check if this model is the best
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = model_name

# Save the best model
if best_model:
    joblib.dump(best_model, 'best_model.pkl')
    print(f"The best model is {best_model_name} with R²: {best_r2:.2f}. It has been saved as 'best_model.pkl'.")

# Compare model performance
results_df = pd.DataFrame(results)

# Plot model performance comparison (R2)
plt.figure(figsize=(12, 8))
plt.bar(results_df['Model'], results_df['R2'], color=['blue', 'green', 'orange', 'purple', 'red', 'cyan', 'magenta'])
plt.title('Model Performance Comparison (R2)')
plt.xlabel('Model')
plt.ylabel('R2 Score')
plt.ylim(0, 1)
plt.grid(True)
plt.savefig('model_performance_comparison_r2.png')
plt.close()

# Plot model performance comparison (Correlation)
plt.figure(figsize=(12, 8))
plt.bar(results_df['Model'], results_df['Correlation'],
        color=['blue', 'green', 'orange', 'purple', 'red', 'cyan', 'magenta'])
plt.title('Model Performance Comparison (Correlation)')
plt.xlabel('Model')
plt.ylabel('Correlation')
plt.ylim(0, 1)
plt.grid(True)
plt.savefig('model_performance_comparison_correlation.png')
plt.close()

# Save results
results_df.to_csv('model_performance_results.csv', index=False)

print("Model comparison and feature importance saved. Check the CSV and plots for details.")
