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
keywords = ["stress during pregnancy", "birth weight", "gestational age", "birth experience"]  # מילות מפתח רלוונטיות

# תאריכים לפני ואחרי תחילת מגפת הקורונה
pre_covid_date = "2019-01-01"  # תאריך התחלתי לתקופה לפני הקורונה
post_covid_date = "2020-03-01"  # תאריך התחלתי לתקופה אחרי פרוץ הקורונה

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

# איסוף פוסטים לפני הקורונה ושמירתם בקובץ נפרד
pre_covid_posts = []
for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    for keyword in keywords:
        pre_covid_posts.extend(collect_posts(subreddit, keyword, "2019-01-01", "2019-12-31", limit=250))

# יצירת DataFrame ושמירתו כ-CSV לפני הקורונה
df_pre_covid = pd.DataFrame(pre_covid_posts)
df_pre_covid.to_csv("pregnancy_birth_outcomes_pre_covid.csv", index=False)
print("Data saved to pregnancy_birth_outcomes_pre_covid.csv")

# איסוף פוסטים במהלך הקורונה ושמירתם בקובץ נפרד
post_covid_posts = []
for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    for keyword in keywords:
        post_covid_posts.extend(collect_posts(subreddit, keyword, "2020-03-01", "2021-12-31", limit=250))

# יצירת DataFrame ושמירתו כ-CSV במהלך הקורונה
df_post_covid = pd.DataFrame(post_covid_posts)
df_post_covid.to_csv("pregnancy_birth_outcomes_post_covid.csv", index=False)
print("Data saved to pregnancy_birth_outcomes_post_covid.csv")
