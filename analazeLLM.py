import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from pingouin import intraclass_corr
import seaborn as sns

# Load data
file_path = 'llm_ready_dataset_labeled.csv'  # Change this to your actual file path
data = pd.read_csv(file_path)

# Columns for analysis
columns = {
    "stress_intensity": {
        "human_1": "Rate the intensity of the stress described in the post on a scale from 1 to 10, where 1 = no stress, 10 = extreme stress, and -1 = no mention of stress. (Danielle)",
        "human_2": "Rate the intensity of the stress described in the post on a scale from 1 to 10, where 1 = no stress, 10 = extreme stress, and -1 = no mention of stress. (Hadas)",
        "model": "Stress Intensity (Model)"
    },
    "emotional_overload": {
        "human_1": "Evaluate the level of emotional overload or mental exhaustion described in the post on a scale from 1 to 10, where 1 = no overload, 10 = extreme overload, and -1 = no mention of overload. (Danielle)",
        "human_2": "Evaluate the level of emotional overload or mental exhaustion described in the post on a scale from 1 to 10, where 1 = no overload, 10 = extreme overload, and -1 = no mention of overload. (Hadas)",
        "model": "Emotional Overload (Model)"
    },
    "pregnancy_week": {
        "human_1": "Identify the week of pregnancy mentioned in the post as a number (e.g., 20 for week 20). If no week is mentioned, return -1. (Danielle)",
        "human_2": "Identify the week of pregnancy mentioned in the post as a number (e.g., 20 for week 20). If no week is mentioned, return -1. (Hadas)",
        "model": "Pregnancy Week (Model)"
    }
}

# Function to calculate correlations and generate results
def calculate_correlations(data, col_info):
    results = {}
    for key, cols in col_info.items():
        human_1 = data[cols["human_1"]]
        human_2 = data[cols["human_2"]]
        model = data[cols["model"]]

        # Handle special case for pregnancy_week
        if key == "pregnancy_week":
            valid_indices = human_1.notna() & human_2.notna() & model.notna()
        else:
            valid_indices = (human_1 != -1) & (human_2 != -1) & (model != -1)

        human_1 = human_1[valid_indices]
        human_2 = human_2[valid_indices]
        model = model[valid_indices]

        # Calculate Pearson and Spearman correlations
        pearson_human, _ = pearsonr(human_1, human_2)
        spearman_human, _ = spearmanr(human_1, human_2)
        pearson_model, _ = pearsonr((human_1 + human_2) / 2, model)
        spearman_model, _ = spearmanr((human_1 + human_2) / 2, model)

        # Prepare ICC data
        icc_data_all = pd.DataFrame({"Rater": np.tile(["Rater1", "Rater2", "Model"], len(human_1)),
                                      "Score": np.concatenate([human_1, human_2, model]),
                                      "Target": np.repeat(range(len(human_1)), 3)})

        icc_data_humans = pd.DataFrame({"Rater": np.tile(["Rater1", "Rater2"], len(human_1)),
                                        "Score": np.concatenate([human_1, human_2]),
                                        "Target": np.repeat(range(len(human_1)), 2)})

        # Calculate ICC for all raters
        try:
            icc_result_all = intraclass_corr(data=icc_data_all, targets="Target", raters="Rater", ratings="Score", nan_policy='omit')
            icc_value_all = icc_result_all[icc_result_all["Type"] == "ICC2k"]["ICC"].values[0]
        except Exception as e:
            print(f"Error calculating ICC for {key} (all): {e}")
            icc_value_all = np.nan

        # Calculate ICC for humans only
        try:
            icc_result_humans = intraclass_corr(data=icc_data_humans, targets="Target", raters="Rater", ratings="Score", nan_policy='omit')
            icc_value_humans = icc_result_humans[icc_result_humans["Type"] == "ICC2k"]["ICC"].values[0]
        except Exception as e:
            print(f"Error calculating ICC for {key} (humans): {e}")
            icc_value_humans = np.nan

        # Save results
        results[key] = {
            "Pearson (Humans)": pearson_human,
            "Spearman (Humans)": spearman_human,
            "Pearson (Humans vs Model)": pearson_model,
            "Spearman (Humans vs Model)": spearman_model,
            "ICC (Humans and Model)": icc_value_all,
            "ICC (Humans Only)": icc_value_humans
        }
    return results

# Generate and save plots
def generate_plots(data, col_info):
    for key, cols in col_info.items():
        plt.figure(figsize=(10, 6))

        # Scatter plot for humans
        sns.scatterplot(x=data[cols["human_1"]], y=data[cols["human_2"]], alpha=0.7)
        plt.title(f"{key.capitalize()} - Human Agreement")
        plt.xlabel("Human 1")
        plt.ylabel("Human 2")
        plt.savefig(f"{key}_human_correlation.png")

        # Scatter plot for humans vs model
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=(data[cols["human_1"]] + data[cols["human_2"]]) / 2, y=data[cols["model"]], alpha=0.7)
        plt.title(f"{key.capitalize()} - Humans vs Model")
        plt.xlabel("Human Average")
        plt.ylabel("Model")
        plt.savefig(f"{key}_human_vs_model_correlation.png")

# Calculate correlations
correlation_results = calculate_correlations(data, columns)

# Save results to Excel
results_df = pd.DataFrame.from_dict(correlation_results, orient='index')
results_df.to_excel('correlation_results.xlsx')

# Generate plots
generate_plots(data, columns)

print("Analysis complete. Results saved to 'correlation_results.xlsx' and plots saved as PNG files.")