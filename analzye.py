import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# טעינת הנתונים
data = pd.read_csv("pregnancy_birth_outcomes.csv")

# דוגמה לניתוח סינון שבועות לידה ולחץ גבוה
preterm_data = data[(data['keyword'] == 'stress during pregnancy') & (data['text'].str.contains('preterm|premature birth|gestational week|week'))]

# חישוב שכיחויות וממוצעים
stress_levels = preterm_data.groupby('stress_level')['gestational_week'].mean()
print("ממוצע שבוע לידה לפי רמת לחץ:", stress_levels)

# גרף עמודות: מספר הלידות המוקדמות לפי רמות לחץ
sns.countplot(data=preterm_data, x='stress_level')
plt.title("מספר הלידות המוקדמות לפי רמת לחץ")
plt.show()
