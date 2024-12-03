import pandas as pd

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


# דוגמה לשימוש
if __name__ == "__main__":
    pre_file_path = "pregnancy_birth_outcomes_pre_blm.csv"  # קובץ ה-PRE
    post_file_path = "pregnancy_birth_outcomes_post_blm.csv"  # קובץ ה-POST
    labeled_file_path = "pregnancy_birth_outcomes_labeling_EXCEL.csv"  # קובץ הלייבלינג
    output_file_path = "pregnancy_birth_outcomes_labeling_EXCEL.csv"  # קובץ לייבלינג מעודכן

    # אינדקסים של הפוסטים הלא רלוונטיים
    irrelevant_indices = [6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 19, 20, 21, 23, 25, 26, 28]

    # החלפת פוסטים לא רלוונטיים
    #replace_irrelevant_posts(pre_file_path, post_file_path, labeled_file_path, output_file_path, irrelevant_indices)
    add_more_pre_posts(pre_file_path, output_file_path, output_file_path, num_posts=65)
