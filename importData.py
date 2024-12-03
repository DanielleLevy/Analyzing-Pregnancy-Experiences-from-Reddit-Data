import praw
import pandas as pd
from datetime import datetime

# התחברות ל-Reddit באמצעות PRAW
reddit = praw.Reddit(
    client_id="BKzAseJNZfeVQrNXiQyu7g",
    client_secret="_fU9KVl4cUlcMPiCnFbjbcR2-VrqYQ",
    user_agent="PregnancyResearchBot by /u/PenPowerful8104"
)

# הגדרת תתי-הפורומים לחיפוש
subreddits = ["BabyBumps", "PregnancyUK", "AskDocs", "NICUParents"]
keywords = [
    "preterm birth", "premature birth", "early delivery",
    "NICU stay", "born early", "gestational age",
    "week of birth", "weeks pregnant", "full term birth", "late delivery",
    "early labor", "premature baby", "preemie", "overdue", "induced labor",
    "c-section delivery", "late pregnancy", "prematurity", "preterm labor",
    "pregnancy stress", "pregnancy anxiety", "mental health during pregnancy",
    "pregnancy depression", "stress during pregnancy", "emotional health",
    "coping during pregnancy", "PTSD pregnancy", "pregnancy trauma"
]

# תאריכי התחלה וסיום חדשים
pre_blm_date = "2019-09-01"  # תאריך התחלה לפני המחאות
blm_start_date = "2020-05-01"  # תחילת תקופת המחאות
blm_end_date = "2020-12-31"  # סיום תקופת המחאות

# פונקציה לספירת מספר המילים בטקסט
def count_words(text):
    return len(text.split())

# פונקציה לאיסוף פוסטים
def collect_posts(subreddit, keyword, start_date, end_date, limit=500, max_words=1000):
    posts = []
    for submission in subreddit.search(keyword, limit=limit, time_filter='all'):
        created_date = datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d')
        if start_date <= created_date <= end_date:
            word_count = count_words(submission.selftext)
            if word_count <= max_words:  # סינון לפי מספר מילים
                posts.append({
                    "subreddit": subreddit.display_name,
                    "title": submission.title,
                    "text": submission.selftext,
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "created": created_date,
                    "keyword": keyword,
                    "word_count": word_count
                })
    return posts

# פונקציה לאיסוף ומיון הפוסטים לפי אורך
def collect_and_sort_posts(subreddits, keywords, start_date, end_date, max_posts=500, max_words=1000):
    all_posts = []
    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        for keyword in keywords:
            all_posts.extend(collect_posts(subreddit, keyword, start_date, end_date, limit=500, max_words=max_words))
    # מיון הפוסטים לפי אורך הטקסט (מהקצר לארוך) ובחירת ה-500 הקצרים ביותר
    sorted_posts = sorted(all_posts, key=lambda x: x["word_count"])
    return sorted_posts[:max_posts]

# איסוף הפוסטים לפני המחאות
pre_blm_posts = collect_and_sort_posts(subreddits, keywords, pre_blm_date, "2020-04-30", max_posts=500, max_words=1000)
print(f"Number of posts collected before BLM: {len(pre_blm_posts)}")
df_pre_blm = pd.DataFrame(pre_blm_posts)
df_pre_blm.to_csv("pregnancy_birth_outcomes_pre_blm.csv", index=False)
print("Data saved to pregnancy_birth_outcomes_pre_blm.csv")

# איסוף הפוסטים במהלך המחאות
post_blm_posts = collect_and_sort_posts(subreddits, keywords, blm_start_date, blm_end_date, max_posts=500, max_words=1000)
print(f"Number of posts collected during BLM: {len(post_blm_posts)}")
df_post_blm = pd.DataFrame(post_blm_posts)
df_post_blm.to_csv("pregnancy_birth_outcomes_post_blm.csv", index=False)
print("Data saved to pregnancy_birth_outcomes_post_blm.csv")
