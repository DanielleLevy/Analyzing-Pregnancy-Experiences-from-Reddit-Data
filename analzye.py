import pandas as pd
# דוגמה לשימוש
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind
# פונקציה למיון קובץ ה-CSV לפי אורך המילים ולהוספת עמודות לשאלות

def process_csv(file_path):
    # קריאת קובץ ה-CSV
    df = pd.read_csv(file_path)

    # הוספת עמודה לספירת מילים בטקסט של הפוסטים
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

    # מיון הפוסטים לפי אורך הטקסט (מהקצר לארוך)
    df = df.sort_values(by='word_count').reset_index(drop=True)

    # רשימת השאלות
    questions = [
        "Rate the intensity of the stress described in the post on a scale from 1 to 10, where 1 = no stress, 10 = extreme stress, and -1 = no mention of stress.",
        "Evaluate the level of emotional overload or mental exhaustion described in the post on a scale from 1 to 10, where 1 = no overload, 10 = extreme overload, and -1 = no mention of overload.",
        "Identify the week of pregnancy mentioned in the post as a number (e.g., 20 for week 20). If no week is mentioned, return -1."
    ]

    # הוספת עמודות לכל שאלה עבור דניאל, הדס והמודל
    for question in questions:
        df[f"{question} (Danielle)"] = ""
        df[f"{question} (Hadas)"] = ""
        df[f"{question} (Model)"] = ""

    # שמירת הקובץ המעודכן לאותו שם קובץ
    df.to_csv(file_path, index=False)
    print(f"Updated CSV saved to {file_path}")
def process_csv_by_score(file_path):
    # קריאת קובץ ה-CSV
    df = pd.read_csv(file_path)

    # מיון הפוסטים לפי SCORE (מהגבוה לנמוך)
    df = df.sort_values(by='score', ascending=False).reset_index(drop=True)

    # שמירת הקובץ המעודכן לאותו שם קובץ
    df.to_csv(file_path, index=False)
    print(f"Updated CSV saved to {file_path}")



def create_labeling_file(pre_file_path, post_file_path, output_file_path):
    # קריאת קבצי ה-PRE וה-POST
    pre_df = pd.read_csv(pre_file_path)
    post_df = pd.read_csv(post_file_path)

    # בדיקה אם עמודות השאלות כבר קיימות והסרתן אם כן
    questions = [
        "Rate the intensity of the stress described in the post on a scale from 1 to 10, where 1 = no stress, 10 = extreme stress, and -1 = no mention of stress.",
        "Evaluate the level of emotional overload or mental exhaustion described in the post on a scale from 1 to 10, where 1 = no overload, 10 = extreme overload, and -1 = no mention of overload.",
        "Identify the week of pregnancy mentioned in the post as a number (e.g., 20 for week 20). If no week is mentioned, return -1."
    ]

    for question in questions:
        for suffix in ["(Danielle)", "(Hadas)", "(Model)"]:
            col_name = f"{question} {suffix}"
            if col_name in pre_df.columns:
                pre_df.drop(columns=[col_name], inplace=True)
            if col_name in post_df.columns:
                post_df.drop(columns=[col_name], inplace=True)

    # סינון פוסטים ללא טקסט והגבלת מילים בין 100 ל-150
    pre_df = pre_df[pre_df['text'].notna() & pre_df['text'].apply(lambda x: 100 <= len(str(x).split()) <= 150)]
    post_df = post_df[post_df['text'].notna() & post_df['text'].apply(lambda x: 100 <= len(str(x).split()) <= 150)]

    # מיון לפי SCORE ובחירת 10 הפוסטים עם הציון הגבוה ביותר
    pre_df = pre_df.sort_values(by='score', ascending=False).head(10)
    post_df = post_df.sort_values(by='score', ascending=False).head(10)

    # הוספת עמודת מקור (PRE או POST)
    pre_df['Source'] = 'PRE'
    post_df['Source'] = 'POST'

    # שילוב הנתונים לקובץ חדש
    combined_df = pd.concat([pre_df, post_df], ignore_index=True)

    # הוספת עמודות לכל שאלה עבור דניאל, הדס והמודל
    for question in questions:
        combined_df[f"{question} (Danielle)"] = ""
        combined_df[f"{question} (Hadas)"] = ""
        combined_df[f"{question} (Model)"] = ""

    # שמירת הקובץ החדש
    combined_df.to_csv(output_file_path, index=False)
    print(f"Labeling file created: {output_file_path}")

def replace_irrelevant_posts(pre_file_path, post_file_path, labeled_file_path, output_file_path, irrelevant_indices):
    # קריאת קבצי ה-PRE וה-POST
    pre_df = pd.read_csv(pre_file_path)
    post_df = pd.read_csv(post_file_path)

    # קריאת קובץ הלייבלינג
    labeled_df = pd.read_csv(labeled_file_path)

    # שמירה על הפוסטים הרלוונטיים בלבד (אלו שלא בתגית "irrelevant_indices")
    relevant_labeled_df = labeled_df.drop(index=irrelevant_indices).reset_index(drop=True)

    # ספירת כמות הפוסטים הנוכחיים מכל מקור
    pre_count = relevant_labeled_df[relevant_labeled_df['Source'] == 'PRE'].shape[0]
    post_count = relevant_labeled_df[relevant_labeled_df['Source'] == 'POST'].shape[0]

    # השלמת פוסטים ל-10 לכל מקור מתוך פוסטים מעבר לראשונים ב-SCORE
    additional_pre_df = pre_df[~pre_df.index.isin(relevant_labeled_df.index)]
    additional_post_df = post_df[~post_df.index.isin(relevant_labeled_df.index)]

    additional_pre_df = additional_pre_df[additional_pre_df['text'].notna() & additional_pre_df['text'].apply(lambda x: 100 <= len(str(x).split()) <= 150)]
    additional_post_df = additional_post_df[additional_post_df['text'].notna() & additional_post_df['text'].apply(lambda x: 100 <= len(str(x).split()) <= 150)]

    additional_pre_df = additional_pre_df.sort_values(by='score', ascending=False).iloc[20:30]
    additional_post_df = additional_post_df.sort_values(by='score', ascending=False).iloc[20:30]

    # הוספת מקור לנתונים החדשים
    additional_pre_df['Source'] = 'PRE'
    additional_post_df['Source'] = 'POST'

    # שילוב הפוסטים הרלוונטיים והחדשים
    combined_df = pd.concat([relevant_labeled_df, additional_pre_df, additional_post_df], ignore_index=True)

    # הוספת עמודות שאלות אם הן אינן קיימות
    questions = [
        "Rate the intensity of the stress described in the post on a scale from 1 to 10, where 1 = no stress, 10 = extreme stress, and -1 = no mention of stress.",
        "Evaluate the level of emotional overload or mental exhaustion described in the post on a scale from 1 to 10, where 1 = no overload, 10 = extreme overload, and -1 = no mention of overload.",
        "Identify the week of pregnancy mentioned in the post as a number (e.g., 20 for week 20). If no week is mentioned, return -1."
    ]

    for question in questions:
        if f"{question} (Danielle)" not in combined_df.columns:
            combined_df[f"{question} (Danielle)"] = ""
        if f"{question} (Hadas)" not in combined_df.columns:
            combined_df[f"{question} (Hadas)"] = ""
        if f"{question} (Model)" not in combined_df.columns:
            combined_df[f"{question} (Model)"] = ""

    # שמירת הקובץ המעודכן
    combined_df.to_csv(output_file_path, index=False)
    print(f"Updated labeled file created: {output_file_path}")
def add_more_pre_posts(pre_file_path, labeled_file_path, output_file_path, num_posts=5):
    # קריאת קובץ ה-PRE
    pre_df = pd.read_csv(pre_file_path)

    # קריאת קובץ הלייבלינג
    labeled_df = pd.read_csv(labeled_file_path)

    # סינון פוסטים שלא נבחרו עדיין
    remaining_pre_df = pre_df[~pre_df.index.isin(labeled_df.index)]

    # סינון לפי טקסט מתאים ומגבלת מילים
    remaining_pre_df = remaining_pre_df[remaining_pre_df['text'].notna() & remaining_pre_df['text'].apply(lambda x: 150 <= len(str(x).split()) <= 180)]

    # הדפסה: כמה פוסטים זמינים לאחר הסינון
    print(f"Number of remaining PRE posts after filtering: {len(remaining_pre_df)}")

    # בחירת פוסטים נוספים מתוך טווח רחב יותר
    additional_pre_df = remaining_pre_df.sort_values(by='score', ascending=False).head(num_posts)
    additional_pre_df['Source'] = 'PRE'

    # הדפסה: כמה פוסטים נבחרו
    print(f"Number of additional PRE posts selected: {len(additional_pre_df)}")

    # הוספת הפוסטים החדשים
    combined_df = pd.concat([labeled_df, additional_pre_df], ignore_index=True)

    # שמירת הקובץ המעודכן
    combined_df.to_csv(output_file_path, index=False)
    print(f"Added {len(additional_pre_df)} new PRE posts to: {output_file_path}")




def analyze_and_plot(file_path, col1, col2, output_plot_path):
    # Load your dataset
    data = pd.read_csv(file_path)

    # Ensure columns are numeric
    data[col1] = pd.to_numeric(data[col1], errors='coerce')
    data[col2] = pd.to_numeric(data[col2], errors='coerce')

    # Drop rows with missing values in the selected columns
    data_cleaned = data.dropna(subset=[col1, col2])

    # Calculate Pearson correlation
    correlation, p_value = pearsonr(data_cleaned[col1], data_cleaned[col2])

    # Print correlation results
    print(f"Pearson correlation coefficient: {correlation}")
    print(f"P-value: {p_value}")
    print(f"Number of data points: {len(data_cleaned)}")

    # Plot the relationship
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data_cleaned[col1], y=data_cleaned[col2], alpha=0.7)
    plt.title("Relationship Between Stress Intensity and Emotional Overload", fontsize=16)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.grid(True)

    # Add legend with statistics
    stats_text = f"Pearson r: {correlation:.2f}\nP-value: {p_value:.2e}\nData Points: {len(data_cleaned)}"
    plt.gca().text(0.95, 0.05, stats_text, fontsize=20, color='black',
                   ha='right', va='bottom', transform=plt.gca().transAxes,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # Save the plot
    plt.savefig(output_plot_path)
    plt.show()

def combine_columns_with_average(file_path, col1, col2, new_col_name):
    # Load your dataset
    data = pd.read_csv(file_path)

    # Ensure columns are numeric
    data[col1] = pd.to_numeric(data[col1], errors='coerce')
    data[col2] = pd.to_numeric(data[col2], errors='coerce')

    # Create a new column with the average of the two columns
    data[new_col_name] = data[[col1, col2]].mean(axis=1)

    # Save the updated dataset in the same file
    data.to_csv(file_path, index=False)
    print(f"Updated dataset with combined column '{new_col_name}' saved to the same file: {file_path}")
def plot_stress_density(file_path, stress_column, period_column, output_density_path):
    # Load your dataset
    data = pd.read_csv(file_path)

    # Ensure relevant columns exist and are numeric
    data[stress_column] = pd.to_numeric(data[stress_column], errors='coerce')
    data[period_column] = data[period_column].astype(str)

    # Drop rows with missing values in the required columns
    data_cleaned = data.dropna(subset=[stress_column, period_column])

    # Plot density
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=data_cleaned, x=stress_column, hue=period_column, fill=True, common_norm=False, alpha=0.5, palette={"PRE": "blue", "POST": "orange"})

    plt.title('Stress Levels Before and During Protests (Density Plot)', fontsize=16)
    plt.xlabel('Stress Level', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(title='Period', labels=['Before Protests', 'During Protests'], fontsize=12)
    plt.grid(True)

    # Save the plot
    plt.savefig(output_density_path)
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind

def analyze_and_plot(file_path, col1, col2, output_plot_path):
    # Load your dataset
    data = pd.read_csv(file_path)

    # Ensure columns are numeric
    data[col1] = pd.to_numeric(data[col1], errors='coerce')
    data[col2] = pd.to_numeric(data[col2], errors='coerce')

    # Drop rows with missing values in the selected columns
    data_cleaned = data.dropna(subset=[col1, col2])

    # Calculate Pearson correlation
    correlation, p_value = pearsonr(data_cleaned[col1], data_cleaned[col2])

    # Print correlation results
    print(f"Pearson correlation coefficient: {correlation}")
    print(f"P-value: {p_value}")
    print(f"Number of data points: {len(data_cleaned)}")

    # Plot the relationship
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data_cleaned[col1], y=data_cleaned[col2], alpha=0.7)
    plt.title("Relationship Between Stress Intensity and Emotional Overload", fontsize=16)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.grid(True)

    # Add legend with statistics
    stats_text = f"Pearson r: {correlation:.2f}\nP-value: {p_value:.2e}\nData Points: {len(data_cleaned)}"
    plt.gca().text(0.95, 0.05, stats_text, fontsize=10, color='black',
                   ha='right', va='bottom', transform=plt.gca().transAxes,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # Save the plot
    plt.savefig(output_plot_path)
    plt.show()

def combine_columns_with_average(file_path, col1, col2, new_col_name):
    # Load your dataset
    data = pd.read_csv(file_path)

    # Ensure columns are numeric
    data[col1] = pd.to_numeric(data[col1], errors='coerce')
    data[col2] = pd.to_numeric(data[col2], errors='coerce')

    # Create a new column with the average of the two columns
    data[new_col_name] = data[[col1, col2]].mean(axis=1)

    # Save the updated dataset in the same file
    data.to_csv(file_path, index=False)
    print(f"Updated dataset with combined column '{new_col_name}' saved to the same file: {file_path}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind

def analyze_and_plot(file_path, col1, col2, output_plot_path):
    # Load your dataset
    data = pd.read_csv(file_path)

    # Ensure columns are numeric
    data[col1] = pd.to_numeric(data[col1], errors='coerce')
    data[col2] = pd.to_numeric(data[col2], errors='coerce')

    # Drop rows with missing values in the selected columns
    data_cleaned = data.dropna(subset=[col1, col2])

    # Calculate Pearson correlation
    correlation, p_value = pearsonr(data_cleaned[col1], data_cleaned[col2])

    # Print correlation results
    print(f"Pearson correlation coefficient: {correlation}")
    print(f"P-value: {p_value}")
    print(f"Number of data points: {len(data_cleaned)}")

    # Plot the relationship
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data_cleaned[col1], y=data_cleaned[col2], alpha=0.7)
    plt.title("Relationship Between Stress Intensity and Emotional Overload", fontsize=16)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.grid(True)

    # Add legend with statistics
    stats_text = f"Pearson r: {correlation:.2f}\nP-value: {p_value:.2e}\nData Points: {len(data_cleaned)}"
    plt.gca().text(0.95, 0.05, stats_text, fontsize=10, color='black',
                   ha='right', va='bottom', transform=plt.gca().transAxes,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # Save the plot
    plt.savefig(output_plot_path)
    plt.show()

def combine_columns_with_average(file_path, col1, col2, new_col_name):
    # Load your dataset
    data = pd.read_csv(file_path)

    # Ensure columns are numeric
    data[col1] = pd.to_numeric(data[col1], errors='coerce')
    data[col2] = pd.to_numeric(data[col2], errors='coerce')

    # Create a new column with the average of the two columns
    data[new_col_name] = data[[col1, col2]].mean(axis=1)

    # Save the updated dataset in the same file
    data.to_csv(file_path, index=False)
    print(f"Updated dataset with combined column '{new_col_name}' saved to the same file: {file_path}")

def plot_stress_density(file_path, stress_column, period_column, output_density_path):
    # Load your dataset
    data = pd.read_csv(file_path)

    # Ensure relevant columns exist and are numeric
    data[stress_column] = pd.to_numeric(data[stress_column], errors='coerce')
    data[period_column] = data[period_column].astype(str)

    # Drop rows with missing values in the required columns
    data_cleaned = data.dropna(subset=[stress_column, period_column])

    # Plot density
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=data_cleaned, x=stress_column, hue=period_column, fill=True, common_norm=False, alpha=0.5, palette={"PRE": "blue", "POST": "orange"})

    plt.title('Stress Levels Before and During Protests (Density Plot)', fontsize=16)
    plt.xlabel('Stress Level', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(title='Period', labels=['Before Protests', 'During Protests'], fontsize=12)

    # Add T-test result to the plot
    t_stat, p_value = ttest_ind(
        data_cleaned[data_cleaned[period_column] == 'PRE'][stress_column],
        data_cleaned[data_cleaned[period_column] == 'POST'][stress_column],
        equal_var=False
    )
    stats_text = f"T-test results:\nT-statistic: {t_stat:.2f}\nP-value: {p_value:.2e}"
    plt.gca().text(0.95, 0.05, stats_text, fontsize=10, color='black',
                   ha='right', va='bottom', transform=plt.gca().transAxes,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.grid(True)

    # Save the plot
    plt.savefig(output_density_path)
    plt.show()




def perform_ttest(file_path, stress_column, period_column):
    # Load your dataset
    data = pd.read_csv(file_path)

    # Ensure relevant columns exist and are numeric
    data[stress_column] = pd.to_numeric(data[stress_column], errors='coerce')
    data[period_column] = data[period_column].astype(str)

    # Drop rows with missing values in the required columns
    data_cleaned = data.dropna(subset=[stress_column, period_column])

    # Separate data into two groups
    pre_data = data_cleaned[data_cleaned[period_column] == 'PRE'][stress_column]
    post_data = data_cleaned[data_cleaned[period_column] == 'POST'][stress_column]

    # Perform t-test
    t_stat, p_value = ttest_ind(pre_data, post_data, equal_var=False)

    # Print results
    print(f"T-test results:\nT-statistic: {t_stat:.2f}\nP-value: {p_value:.2e}")

    # Visualize results
    plt.figure(figsize=(8, 5))
    sns.barplot(x=["Before Protests", "During Protests"], y=[pre_data.mean(), post_data.mean()], palette=["blue", "orange"])
    plt.title("Average Stress Levels Before and During Protests", fontsize=16)
    plt.ylabel("Average Stress Level", fontsize=14)
    plt.xlabel("Period", fontsize=14)
    plt.text(0, pre_data.mean(), f"Mean: {pre_data.mean():.2f}", ha='center', va='bottom', fontsize=12)
    plt.text(1, post_data.mean(), f"Mean: {post_data.mean():.2f}", ha='center', va='bottom', fontsize=12)

    # Add T-test result
    plt.figtext(0.5, -0.1, f"T-test: T-statistic = {t_stat:.2f}, P-value = {p_value:.2e}", wrap=True, horizontalalignment='center', fontsize=12)

    plt.grid(axis='y')

    # Save the bar plot
    plt.savefig("ttest_stress_comparison.png")
    plt.show()

if __name__ == "__main__":
    # Define file path and columns to analyze
    file_path = 'llm_ready_dataset_labeled.csv'
    col1 = 'Stress Intensity (Model)'
    col2 = 'Emotional Overload (Model)'
    output_plot_path = "ex3graphs/scatter_plot_stress_vs_emotional_overload.png"

    # Call the analysis function
    #analyze_and_plot(file_path, col1, col2, output_plot_path)
# Combine columns with average
    # Combine columns with average
    new_col_name = 'Average Stress and Emotional Overload'
    #combine_columns_with_average(file_path, col1, col2, new_col_name)
    # Plot histogram for stress levels by period
    stress_column = 'Average Stress and Emotional Overload'
    period_column = 'Source'
    output_histogram_path = "ex3graphs/stress_histogram_by_period.png"
    plot_stress_density(file_path, stress_column, period_column, output_histogram_path)
    # Perform T-test for stress levels by period
    perform_ttest(file_path, stress_column, period_column)