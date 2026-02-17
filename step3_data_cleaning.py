import pandas as pd

print("🔄 Loading raw dataset...")

# Load raw dataset
df = pd.read_csv("cleaned_career_recommender.csv", encoding="latin1")

print("Original Shape:", df.shape)

# ============================================
# Basic Cleaning
# ============================================

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Standardize column names
df.columns = df.columns.str.strip().str.lower()
df.columns = df.columns.str.replace(" ", "_")


# Handle missing values
df.fillna("unknown", inplace=True)

# ============================================
# Standardize Education Levels
# ============================================

education_mapping = {
    "10th": "10th",
    "ssc": "10th",
    "12th": "12th",
    "intermediate": "12th",
    "diploma": "Diploma",
    "btech": "Undergraduate",
    "bachelor": "Undergraduate",
    "ug": "Undergraduate",
    "mtech": "Postgraduate",
    "masters": "Postgraduate",
    "pg": "Postgraduate",
    "working": "Working Professional"
}

if "education_level" in df.columns:
    df["education_level"] = df["education_level"].replace(education_mapping)

# ============================================
# Standardize Stream
# ============================================

stream_mapping = {
    "science": "Science",
    "commerce": "Commerce",
    "arts": "Arts",
    "technical": "Technical",
    "non technical": "Non-Technical"
}

if "stream" in df.columns:
    df["stream"] = df["stream"].str.lower().replace(stream_mapping)

# ============================================
# Convert Age & Experience
# ============================================

if "age" in df.columns:
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(0).astype(int)

if "years_experience" in df.columns:
    df["years_experience"] = pd.to_numeric(
        df["years_experience"], errors="coerce"
    ).fillna(0).astype(int)

# ============================================
# Clean Text Columns
# ============================================

text_columns = [
    "ug_course",
    "ug_specialization",
    "current_job",
    "interests",
    "skills",
    "preferred_work_type",
    "career_gap",
    "first_job_title"
]

for col in text_columns:
    if col in df.columns:
        df[col] = df[col].str.lower().str.strip()

# ============================================
# Save Cleaned Dataset
# ============================================

df.to_csv("cleaned_career_recommender.csv", index=False)

print("✅ Data cleaning completed successfully!")
print("Cleaned Shape:", df.shape)
print("File saved as cleaned_career_recommender.csv")
