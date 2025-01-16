# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau

# Load the dataset
df = pd.read_csv('llm_ready_dataset_labeled_with_sentiment.csv')

# Clean the dataset
relevant_columns = ['text', 'Pregnancy Week (Model)', 'neg', 'neu', 'pos', 'compound', 'Average Stress and Emotional Overload']
df_cleaned = df.dropna(subset=relevant_columns)

# Feature engineering
df_cleaned['text_length'] = df_cleaned['text'].apply(len)
df_cleaned['question_marks'] = df_cleaned['text'].str.count('\?')
df_cleaned['neg_pos_ratio'] = df_cleaned['neg'] / (df_cleaned['pos'] + 1e-5)
df_cleaned['week_neg_interaction'] = df_cleaned['Pregnancy Week (Model)'] * df_cleaned['neg']

# Scatter plot for Pregnancy Week vs. Stress
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Pregnancy Week (Model)', y='Average Stress and Emotional Overload', data=df_cleaned, alpha=0.7)
plt.title('Scatter Plot: Pregnancy Week vs. Average Stress and Emotional Overload')
plt.xlabel('Pregnancy Week (Model)')
plt.ylabel('Average Stress and Emotional Overload')
plt.savefig('Pregnancy_Week_vs_Stress_Scatter.png')
plt.show()

# Box plot for Pregnancy Week Buckets
plt.figure(figsize=(12, 6))
df_cleaned['Pregnancy Week (Buckets)'] = pd.cut(df_cleaned['Pregnancy Week (Model)'], bins=5, labels=['1-20', '21-40', '41-60', '61-80', '81+'])
sns.boxplot(x='Pregnancy Week (Buckets)', y='Average Stress and Emotional Overload', data=df_cleaned)
plt.title('Box Plot: Stress Distribution Across Pregnancy Weeks')
plt.xlabel('Pregnancy Week Buckets')
plt.ylabel('Average Stress and Emotional Overload')
plt.savefig('Stress_Distribution_Across_Pregnancy_Weeks.png')
plt.show()

# Correlation analysis for all features
features = ['Pregnancy Week (Model)', 'neg', 'neu', 'pos', 'compound', 'text_length', 'question_marks', 'neg_pos_ratio', 'week_neg_interaction']
for feature in features:
    print(f"Analyzing feature: {feature}")

    # Scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=feature, y='Average Stress and Emotional Overload', data=df_cleaned, alpha=0.7)
    plt.title(f'Scatter Plot: {feature} vs. Average Stress and Emotional Overload')
    plt.xlabel(feature)
    plt.ylabel('Average Stress and Emotional Overload')
    plt.savefig(f'{feature}_vs_Stress_Scatter.png')
    plt.show()

    # Pearson correlation
    pearson_corr = df_cleaned[feature].corr(df_cleaned['Average Stress and Emotional Overload'])
    print(f"Pearson Correlation Coefficient for {feature}: {pearson_corr:.2f}")

    # Spearman correlation
    spearman_corr, spearman_pval = spearmanr(df_cleaned[feature], df_cleaned['Average Stress and Emotional Overload'])
    print(f"Spearman Correlation Coefficient for {feature}: {spearman_corr:.2f}, P-value: {spearman_pval:.4f}")

    # Kendall correlation
    kendall_corr, kendall_pval = kendalltau(df_cleaned[feature], df_cleaned['Average Stress and Emotional Overload'])
    print(f"Kendall Correlation Coefficient for {feature}: {kendall_corr:.2f}, P-value: {kendall_pval:.4f}")

    # Interpretation of results
    if spearman_pval < 0.05:
        print(f"The Spearman correlation for {feature} is statistically significant.")
    else:
        print(f"The Spearman correlation for {feature} is not statistically significant.")

    if kendall_pval < 0.05:
        print(f"The Kendall correlation for {feature} is statistically significant.")
    else:
        print(f"The Kendall correlation for {feature} is not statistically significant.")

    print("\n")
