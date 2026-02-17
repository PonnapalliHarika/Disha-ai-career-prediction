import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("🔄 Loading dataset...")

# ============================================
# Load Dataset
# ============================================

df = pd.read_csv("cleaned_career_recommender.csv")
df.fillna("unknown", inplace=True)
df = pd.read_csv("cleaned_career_recommender.csv", encoding="latin1")
print("COLUMNS ARE:")
print(df.columns)
print("Shape:", df.shape)

# ============================================
# Create Combined Features (MUST MATCH step4)
# ============================================

# Ensure required columns exist
required_columns = [
    "age", "education_level", "stream", "ug_course",
    "ug_specialization", "current_job", "years_experience",
    "interests", "skills", "preferred_work_type",
    "career_gap"
]

for col in required_columns:
    if col not in df.columns:
        df[col] = "unknown"

# Convert numeric columns safely
df["age"] = df["age"].astype(str)
df["years_experience"] = df["years_experience"].astype(str)

# Create combined features
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

print("✅ Combined features ready")

# ============================================
# Load Vectorizer (Created in step4)
# ============================================

vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

X = vectorizer.transform(df["combined_features"])
y = df["first_job_title"]
# Remove rare job titles (less than 3 samples)
value_counts = y.value_counts()
valid_classes = value_counts[value_counts >= 3].index

df = df[df["first_job_title"].isin(valid_classes)]

# Recreate X and y after filtering
X = vectorizer.transform(df["combined_features"])
y = df["first_job_title"]

print("Remaining classes:", y.nunique())
print("Remaining samples:", len(y))


print("✅ Feature transformation complete")

# ============================================
# Train-Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ============================================
# Random Forest + Hyperparameter Tuning
# ============================================

rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20, 40],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2
)

print("🚀 Training Random Forest...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# ============================================
# Evaluation
# ============================================

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n✅ Model trained successfully")
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ============================================
# Save Model
# ============================================

pickle.dump(best_model, open("career_model.pkl", "wb"))

print("🎉 Random Forest Model Saved Successfully")
