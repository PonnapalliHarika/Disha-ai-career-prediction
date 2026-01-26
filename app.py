import streamlit as st
import pickle
import pandas as pd
from roadmap import get_roadmap


# Load trained model and vectorizer
model = pickle.load(open("career_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Disha AI", page_icon="🎯")

st.title("🎯 Disha AI – Career Path Prediction")
st.write("Get personalized career recommendations based on your profile")

# User Inputs
ug_course = st.text_input("Your UG Course")
ug_specialization = st.text_input("Your Specialization")
interests = st.text_area("Your Interests")
skills = st.text_area("Your Skills")

if st.button("Predict Career"):
    if ug_course and ug_specialization and interests and skills:
        user_input = ug_course + " " + ug_specialization + " " + interests + " " + skills
        user_vector = vectorizer.transform([user_input])

        # Probabilities
        probabilities = model.predict_proba(user_vector)[0]
        careers = model.classes_

        results = sorted(
            zip(careers, probabilities),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        st.subheader("🔮 Top Career Predictions")
        for career, prob in results:
            st.write(f"**{career}** → {round(prob*100, 2)}%")

        # Roadmap for top career
        top_career = results[0][0]
        st.subheader(f"🛣 Roadmap for {top_career}")
        roadmap = get_roadmap(top_career)

        for step in roadmap:
            st.write("✅", step)

    else:
        st.warning("⚠️ Please fill all fields")
