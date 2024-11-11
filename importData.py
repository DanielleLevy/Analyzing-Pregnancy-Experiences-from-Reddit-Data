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
subreddits = ["BabyBumps", "PregnancyUK", "AskDocs"]  # תתי-פורומים ספציפיים
keywords = ["stress during pregnancy", "birth weight", "gestational age", "birth experience"]  # מילות מפתח רלוונטיות

# איסוף פוסטים
posts = []
for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    for keyword in keywords:
        for submission in subreddit.search(keyword, limit=500):  # את יכולה להגדיל את ה-limit לפי הצורך
            posts.append({
                "subreddit": subreddit_name,
                "title": submission.title,
                "text": submission.selftext,
                "score": submission.score,
                "num_comments": submission.num_comments,
                "created": datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                "keyword": keyword
            })

# יצירת DataFrame ושמירתו כ-CSV
df = pd.DataFrame(posts)
df.to_csv("pregnancy_birth_outcomes.csv", index=False)
print("Data saved to pregnancy_birth_outcomes.csv")