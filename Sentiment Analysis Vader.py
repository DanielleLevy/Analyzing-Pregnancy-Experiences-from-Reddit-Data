import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from nltk.util import ngrams


def load_and_preprocess_data(file_path):
    # Load the data
    data = pd.read_csv(file_path)

    # Ensure necessary columns exist
    if 'text' not in data.columns or 'Average Stress and Emotional Overload' not in data.columns:
        raise ValueError("The dataset must contain 'text' and 'Average Stress and Emotional Overload' columns.")

    return data


def analyze_sentiment_vader(data):
    # Download the necessary resource for VADER
    download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

    # Apply VADER analysis to each text entry
    sentiments = data['text'].apply(sia.polarity_scores)
    sentiment_df = pd.DataFrame(list(sentiments))

    # Combine the original data with sentiment scores
    data = pd.concat([data, sentiment_df], axis=1)

    return data


def calculate_statistics(data):
    # Pearson correlation
    pos_corr, pos_pval = pearsonr(data['Average Stress and Emotional Overload'], data['pos'])
    neg_corr, neg_pval = pearsonr(data['Average Stress and Emotional Overload'], data['neg'])

    # Format results
    stats_text = (
        f"Positive Sentiment Correlation: r={pos_corr:.2f}, p={pos_pval:.2e}\n"
        f"Negative Sentiment Correlation: r={neg_corr:.2f}, p={neg_pval:.2e}"
    )
    return stats_text


def plot_sentiment_vs_stress(data, sentiment_column, stress_column, output_file):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=data, x=stress_column, y=sentiment_column, alpha=0.6)
    sns.regplot(data=data, x=stress_column, y=sentiment_column, scatter=False, color='red')

    # Add Pearson correlation to the plot
    corr, pval = pearsonr(data[stress_column], data[sentiment_column])
    plt.text(0.05, 0.95, f"r={corr:.2f}, p={pval:.2e}", transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.title(f'Relationship between {sentiment_column.capitalize()} and {stress_column}', fontsize=16)
    plt.xlabel(stress_column, fontsize=14)
    plt.ylabel(sentiment_column.capitalize(), fontsize=14)
    plt.grid(True)
    plt.savefig(output_file)
    plt.show()


def analyze_tfidf(data, text_column, output_file):
    vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data[text_column])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # Summarize the TF-IDF scores
    tfidf_scores = tfidf_df.sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=tfidf_scores.values, y=tfidf_scores.index, palette='viridis')
    plt.title('Top 20 TF-IDF Words', fontsize=16)
    plt.xlabel('TF-IDF Score', fontsize=14)
    plt.ylabel('Words', fontsize=14)
    plt.grid(True)
    plt.savefig(output_file)
    plt.show()


def analyze_ngrams(data, text_column, n, output_file):
    # Generate n-grams and count their frequencies
    all_text = ' '.join(data[text_column].dropna().values)
    ngram_counts = Counter(ngrams(all_text.split(), n))
    most_common = ngram_counts.most_common(20)

    # Prepare data for plotting
    ngrams_list, counts = zip(*most_common)
    ngrams_strings = [' '.join(ngram) for ngram in ngrams_list]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts, y=ngrams_strings, palette='coolwarm')
    plt.title(f'Top {n}-grams', fontsize=16)
    plt.xlabel('Frequency', fontsize=14)
    plt.ylabel('N-grams', fontsize=14)
    plt.grid(True)
    plt.savefig(output_file)
    plt.show()


def analyze_similarity_by_stress(data, text_column, stress_column, output_file):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data[text_column].dropna())
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Calculate average similarity for each stress category
    data['Stress_Category'] = pd.cut(data[stress_column], bins=[0, 3, 6, 10], labels=['Low', 'Medium', 'High'],
                                     include_lowest=True)
    avg_similarities = []
    categories = data['Stress_Category'].unique()
    for category in categories:
        indices = data[data['Stress_Category'] == category].index
        avg_similarity = similarity_matrix[indices][:, indices].mean()
        avg_similarities.append(avg_similarity)

    # Plot average similarities
    plt.figure(figsize=(8, 6))
    sns.barplot(x=categories, y=avg_similarities, palette='magma')
    plt.title('Average Text Similarity by Stress Levels', fontsize=16)
    plt.xlabel('Stress Levels', fontsize=14)
    plt.ylabel('Average Similarity', fontsize=14)
    plt.grid(True)
    plt.savefig(output_file)
    plt.show()


def analyze_punctuation(data, text_column, output_file):
    punctuation_counts = {
        'Exclamation Marks': data[text_column].str.count('!').sum(),
        'Question Marks': data[text_column].str.count('\?').sum(),
        'Emojis': data[text_column].str.count(r'[\U0001F600-\U0001F64F]').sum()
    }

    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(punctuation_counts.values()), y=list(punctuation_counts.keys()), palette='plasma')
    plt.title('Punctuation and Emoji Usage', fontsize=16)
    plt.xlabel('Counts', fontsize=14)
    plt.ylabel('Type', fontsize=14)
    plt.grid(True)
    plt.savefig(output_file)
    plt.show()


def main(file_path):
    # Load and preprocess data
    data = load_and_preprocess_data(file_path)

    # Analyze sentiment using VADER
    data_with_sentiment = analyze_sentiment_vader(data)

    # Save the updated dataset
    output_file = file_path.replace('.csv', '_with_sentiment.csv')
    data_with_sentiment.to_csv(output_file, index=False)
    print(f"Sentiment analysis completed. Data saved to {output_file}")

    # Analyze and plot TF-IDF
    analyze_tfidf(data, 'text', 'tfidf_top_words.png')

    # Analyze and plot bigrams
    analyze_ngrams(data, 'text', 2, 'bigrams_top.png')

    # Analyze and plot trigram frequency
    analyze_ngrams(data, 'text', 3, 'trigrams_top.png')

    # Analyze similarity by stress levels
    analyze_similarity_by_stress(data, 'text', 'Average Stress and Emotional Overload', 'similarity_by_stress.png')

    # Analyze punctuation and emoji usage
    analyze_punctuation(data, 'text', 'punctuation_and_emoji_usage.png')


# Example usage
if __name__ == "__main__":
    file_path = 'llm_ready_dataset_labeled.csv'  # Replace with your actual file path
    main(file_path)
