import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download


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


def plot_sentiment_vs_stress(data, sentiment_column, stress_column, output_file):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=data, x=stress_column, y=sentiment_column, alpha=0.6)
    sns.regplot(data=data, x=stress_column, y=sentiment_column, scatter=False, color='red')
    plt.title(f'Relationship between {sentiment_column.capitalize()} and {stress_column}', fontsize=16)
    plt.xlabel(stress_column, fontsize=14)
    plt.ylabel(sentiment_column.capitalize(), fontsize=14)
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

    # Plot and save relationships between sentiment and stress level
    plot_sentiment_vs_stress(data_with_sentiment, 'pos', 'Average Stress and Emotional Overload',
                             'positive_vs_stress.png')
    plot_sentiment_vs_stress(data_with_sentiment, 'neg', 'Average Stress and Emotional Overload',
                             'negative_vs_stress.png')


# Example usage
if __name__ == "__main__":
    file_path = 'llm_ready_dataset_labeled.csv'  # Replace with your actual file path
    main(file_path)
