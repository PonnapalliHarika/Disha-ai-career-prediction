import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load cleaned dataset
df = pd.read_csv("cleaned_career_recommender.csv")

# Handle missing values
df.fillna("unknown", inplace=True)

# Combine important text columns
df["combined_features"] = (
    df["ug_course"] + " " +
    df["ug_specialization"] + " " +
    df["interests"] + " " +
    df["skills"]
)

# Convert text to numerical features
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["combined_features"])

# Target variable
y = df["first_job_title"]

print("Step 4 completed successfully ✅")
print("Feature matrix shape:", X.shape)
print("Sample predictions target:")
print(y.head())
