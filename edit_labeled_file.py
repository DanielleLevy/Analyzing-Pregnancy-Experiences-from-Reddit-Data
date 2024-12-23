import pandas as pd

# Load the dataset
file_path = 'llm_ready_dataset_labeled.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)


# Function to extract and update values from "LLM Response"
def update_model_columns(row):
    if (row['Stress Intensity (Model)'] == -1 and
            row['Emotional Overload (Model)'] == -1 and
            row['Pregnancy Week (Model)'] == -1):

        # Parse the LLM Response
        response = str(row['LLM Response']).split("\n")
        try:
            intensity = float(response[0].split(".")[1].strip())
            overload = float(response[1].split(".")[1].strip())
            week = float(response[2].split(".")[1].strip())

            # Update the model columns
            row['Stress Intensity (Model)'] = intensity
            row['Emotional Overload (Model)'] = overload
            row['Pregnancy Week (Model)'] = week
        except (IndexError, ValueError):
            pass  # Ignore rows where response is not well-formed
    return row


# Apply the function to update the data
data = data.apply(update_model_columns, axis=1)

# Save the updated dataset
data.to_csv('llm_ready_dataset_labeled.csv', index=False)

print("The dataset has been updated and saved as 'updated_dataset.csv'.")
