import pandas as pd
# Function to add unique posts from a dataset (PRE or POST) to the labeled file
# פונקציה להוספת פוסטים ייחודיים מקובץ מקור (PRE או POST) לקובץ הלייבלינג
# Function to add unique posts from a dataset (PRE or POST) to the labeled file
# פונקציה להוספת פוסטים ייחודיים מקובץ מקור (PRE או POST) לקובץ הלייבלינג
# פונקציה להוספת פוסטים ייחודיים מקובץ מקור (PRE או POST) לקובץ הלייבלינג
def add_unique_posts(source_file_path, labeled_file_path, source_label, num_posts):
    # קריאה של קובץ המקור (PRE או POST)
    source_df = pd.read_csv(source_file_path)

    # קריאה של קובץ הלייבלינג
    labeled_df = pd.read_csv(labeled_file_path)

    # רשימת טקסטים שכבר קיימים בקובץ הלייבלינג
    used_texts = set(labeled_df['text'].dropna())

    # סינון קפדני של פוסטים ייחודיים בלבד
    remaining_source_df = source_df[~source_df['text'].isin(used_texts)]

    # סינון לפי טקסט תקין וטווח מילים מתאים
    remaining_source_df = remaining_source_df[
        remaining_source_df['text'].notna() &
        remaining_source_df['text'].apply(lambda x: 450 <= len(str(x).split()) <= 500)
    ]

    # העדפת פוסטים מתוך BabyBumps ו-PregnancyUK
    babybumps_posts = remaining_source_df[remaining_source_df['subreddit'] == 'BabyBumps']
    pregnancyuk_posts = remaining_source_df[remaining_source_df['subreddit'] == 'PregnancyUK']
    other_posts = remaining_source_df[~remaining_source_df['subreddit'].isin(['BabyBumps', 'PregnancyUK'])]

    # שילוב הפוסטים תוך העדפת BabyBumps ו-PregnancyUK
    remaining_source_df = pd.concat([babybumps_posts, pregnancyuk_posts, other_posts])

    # בחירת פוסטים נוספים לפי הציון הגבוה ביותר
    additional_source_df = remaining_source_df.sort_values(by='score', ascending=False).head(num_posts)
    additional_source_df['Source'] = source_label

    # הוספת הפוסטים החדשים לקובץ הלייבלינג
    combined_df = pd.concat([labeled_df, additional_source_df], ignore_index=True)

    # שמירת הקובץ המעודכן
    combined_df.to_csv(labeled_file_path, index=False)
    print(f"Added {len(additional_source_df)} new {source_label} posts to: {labeled_file_path}")




# Function to add more posts from a dataset (PRE or POST) to the labeled file
def add_more_posts(source_file_path, labeled_file_path, output_file_path, source_label, num_posts=5):
    # Load the source dataset (PRE or POST)
    source_df = pd.read_csv(source_file_path)

    # Load the labeled dataset
    labeled_df = pd.read_csv(labeled_file_path)

    # Filter posts that haven't been used yet
    remaining_source_df = source_df[~source_df.index.isin(labeled_df.index)]

    # Apply text and word count filters
    remaining_source_df = remaining_source_df[remaining_source_df['text'].notna() & remaining_source_df['text'].apply(lambda x: 150 <= len(str(x).split()) <= 250)]

    # Select top posts based on score
    additional_source_df = remaining_source_df.sort_values(by='score', ascending=False).head(num_posts)
    additional_source_df['Source'] = source_label

    # Add new posts to the labeled dataset
    combined_df = pd.concat([labeled_df, additional_source_df], ignore_index=True)

    # Save the updated dataset
    combined_df.to_csv(output_file_path, index=False)
    print(f"Added {len(additional_source_df)} new {source_label} posts to: {output_file_path}")

# Define file paths
pre_file_path = 'pregnancy_birth_outcomes_pre_blm.csv'   # Path to "PRE" dataset
post_file_path = 'pregnancy_birth_outcomes_post_blm.csv' # Path to "POST" dataset
labeled_file_path = 'final_balanced_dataset.csv'          # Path to save the final balanced dataset
# Load the labeled dataset
cleaned_df = pd.read_csv(labeled_file_path)

# Calculate how many more posts are needed (doubling the quantity for filtering)
current_pre_count = cleaned_df[cleaned_df['Source'] == 'PRE'].shape[0]
current_post_count = cleaned_df[cleaned_df['Source'] == 'POST'].shape[0]
num_additional_pre_posts = (50 - current_pre_count) * 4
num_additional_post_posts = (50 - current_post_count) * 4

# Add more unique PRE posts if necessary
if num_additional_pre_posts > 0:
    add_unique_posts(pre_file_path, labeled_file_path, source_label='PRE', num_posts=num_additional_pre_posts)

# Add more unique POST posts if necessary
if num_additional_post_posts > 0:
    add_unique_posts(post_file_path, labeled_file_path, source_label='POST', num_posts=num_additional_post_posts)

print("The updated dataset now includes additional posts for further filtering.")
