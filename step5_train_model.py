import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load cleaned dataset
df = pd.read_csv("cleaned_career_recommender.csv")
df.fillna("unknown", inplace=True)

# Combine text features
df["combined_features"] = (
    df["ug_course"] + " " +
    df["ug_specialization"] + " " +
    df["interests"] + " " +
    df["skills"]
)

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["combined_features"])
y = df["first_job_title"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model trained successfully ✅")
print("Accuracy:", accuracy)

# Save model and vectorizer
pickle.dump(model, open("career_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and vectorizer saved 🎉")
