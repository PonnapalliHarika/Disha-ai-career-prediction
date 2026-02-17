import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

print("🔄 Loading dataset...")

# Load cleaned dataset
df = pd.read_csv("DISHA_AI/cleaned_career_recommender.csv")


# Handle missing values
df.fillna("unknown", inplace=True)

# Convert numeric columns safely
if "age" in df.columns:
    df["age"] = df["age"].astype(str)
else:
    df["age"] = "unknown"

if "years_experience" in df.columns:
    df["years_experience"] = df["years_experience"].astype(str)
else:
    df["years_experience"] = "0"

# Ensure required columns exist
required_columns = [
    "education_level",
    "stream",
    "ug_course",
    "ug_specialization",
    "current_job",
    "interests",
    "skills",
    "preferred_work_type",
    "career_gap"
]

for col in required_columns:
    if col not in df.columns:
        df[col] = "unknown"

# ============================================
# Create Combined Features with Prefix Tags
# ============================================

df["combined_features"] = (
    "age_" + df["age"] + " " +
    "edu_" + df["education_level"] + " " +
    "stream_" + df["stream"] + " " +
    "course_" + df["ug_course"] + " " +
    "spec_" + df["ug_specialization"] + " " +
    "job_" + df["current_job"] + " " +
    "exp_" + df["years_experience"] + " " +
    "interest_" + df["interests"] + " " +
    "skill_" + df["skills"] + " " +
    "work_" + df["preferred_work_type"] + " " +
    "gap_" + df["career_gap"]
)

print("✅ Combined features created")

# ============================================
# TF-IDF Vectorization
# ============================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=8000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X = vectorizer.fit_transform(df["combined_features"])
y = df["first_job_title"]

print("✅ TF-IDF transformation complete")
print("Feature matrix shape:", X.shape)
print("Number of target classes:", y.nunique())

# ============================================
# Save Vectorizer
# ============================================

pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("🎉 Feature engineering completed successfully!")
print("Vectorizer saved as vectorizer.pkl")

