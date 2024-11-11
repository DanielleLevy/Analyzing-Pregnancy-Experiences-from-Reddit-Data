import praw
import pandas as pd
from datetime import datetime

#Danielle
# התחברות ל-Reddit באמצעות PRAW
reddit = praw.Reddit(
    client_id="BKzAseJNZfeVQrNXiQyu7g",  # הכניסי כאן את ה-Client ID שלך
    client_secret="_fU9KVl4cUlcMPiCnFbjbcR2-VrqYQ",  # הכניסי כאן את ה-Client Secret שלך
    user_agent="PregnancyResearchBot by /u/PenPowerful8104"  # ודאי שה-username הוא שם המשתמש שלך ב-Reddit
)
"""
#Hadas
reddit = praw.Reddit(
    client_id="-BgnVREvza43_vK7ALi2nQ",  # הכניסי כאן את ה-Client ID שלך
    client_secret="rygbTMY5nH1mGbQmPdrzSOD_tqbwPg",  # הכניסי כאן את ה-Client Secret שלך
    user_agent="PregnancyResearchBot by /u/CompetitionGrand9590"  # ודאי שה-username הוא שם המשתמש שלך ב-Reddit
)
"""
# הגדרת תתי-הפורומים לחיפוש
subreddits = ["BabyBumps", "PregnancyUK", "AskDocs", "NICUParents"]  # תתי-פורומים רלוונטיים
# מילות מפתח שמתמקדות בשבוע הלידה ובסיכון ללידה מוקדמת או מאוחרת
keywords = [
    "preterm birth", "premature birth", "early delivery",
    "NICU stay", "born early", "gestational age",
    "week of birth", "weeks pregnant", "full term birth", "late delivery"
]

# הגדרת התקופות (לפני המחאות ובמהלך המחאות)
pre_blm_date = "2020-01-01"  # תקופה שלפני המחאות
blm_start_date = "2020-05-01"  # תחילת המחאות
blm_end_date = "2020-12-31"  # סיום המחאות

# פונקציה לאיסוף פוסטים בתקופות שונות
def collect_posts(subreddit, keyword, start_date, end_date, limit=500):
    posts = []
    for submission in subreddit.search(keyword, limit=limit, time_filter='all'):
        created_date = datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d')
        if start_date <= created_date <= end_date:
            posts.append({
                "subreddit": subreddit.display_name,
                "title": submission.title,
                "text": submission.selftext,
                "score": submission.score,
                "num_comments": submission.num_comments,
                "created": created_date,
                "keyword": keyword
            })
    return posts

# איסוף פוסטים לפני המחאות ושמירתם בקובץ נפרד
pre_blm_posts = []
for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    for keyword in keywords:
        pre_blm_posts.extend(collect_posts(subreddit, keyword, pre_blm_date, "2020-04-30", limit=250))

# יצירת DataFrame ושמירתו כ-CSV לפני המחאות
df_pre_blm = pd.DataFrame(pre_blm_posts)
df_pre_blm.to_csv("pregnancy_birth_outcomes_pre_blm.csv", index=False)
print("Data saved to pregnancy_birth_outcomes_pre_blm.csv")

# איסוף פוסטים במהלך המחאות ושמירתם בקובץ נפרד
post_blm_posts = []
for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    for keyword in keywords:
        post_blm_posts.extend(collect_posts(subreddit, keyword, blm_start_date, blm_end_date, limit=250))

# יצירת DataFrame ושמירתו כ-CSV במהלך המחאות
df_post_blm = pd.DataFrame(post_blm_posts)
df_post_blm.to_csv("pregnancy_birth_outcomes_post_blm.csv", index=False)
print("Data saved to pregnancy_birth_outcomes_post_blm.csv")