import streamlit as st
import pandas as pd
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# STREAM RELATION MAPPING
# ----------------------------

STREAM_MAP = {
    "SCIENCE": ["CSE", "ECE", "EEE"],
    "MATHS": ["CSE", "ACCOUNTS", "COMMERCE"],
    "DESIGN": ["ARTS", "CSE"],
    "ACCOUNTS": ["COMMERCE"],
    "ARTS": ["DESIGN", "COMMERCE"],
    "COMMERCE": ["ACCOUNTS", "MANAGEMENT"],
    "CSE": ["ECE"],
    "ECE": ["EEE"],
    "EEE": ["ECE"],
    "CIVIL": ["MECHANICAL"],
}


# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="AI Career Path Engine",
    page_icon="🎯",
    layout="wide"
)




# ----------------------------
# HEADER
# ----------------------------
st.markdown("""
<h1>🎓 DISHA AI - Career Path Prediction</h1>
<p style="text-align:center;color:#aaa;">
Personalized | Data-Driven | Smart Guidance
</p>
""", unsafe_allow_html=True)


# ----------------------------
# LOAD DATA
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "career_dataset.csv")

df = pd.read_csv(file_path)
df.fillna("", inplace=True)


# ----------------------------
# COMBINE FEATURES
# ----------------------------
df["combined"] = (
    df["Title"] + " " +
    df["Description"] + " " +
    df["Skills"] + " " +
    df["Technology Skills"]
)

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["combined"])


# ----------------------------
# SESSION STATE INIT
# ----------------------------
if "step" not in st.session_state:
    st.session_state.step = 1


# ----------------------------
# RECOMMENDATION ENGINE
# ----------------------------
def recommend(user_input, stream):

    df["Stream_clean"] = df["Stream"].str.upper().str.strip()
    user_stream = stream.upper().strip()

    # Strict match
    filtered_df = df[df["Stream_clean"] == user_stream]

    # Related streams
    if filtered_df.empty and user_stream in STREAM_MAP:
        related = STREAM_MAP[user_stream]
        filtered_df = df[df["Stream_clean"].isin(related)]

    # Fallback
    if filtered_df.empty:
        filtered_df = df.copy()

    X_filtered = vectorizer.transform(filtered_df["combined"])
    user_vec = vectorizer.transform([user_input])

    similarity = cosine_similarity(user_vec, X_filtered)[0]

    top_indices = similarity.argsort()[-3:][::-1]

    results = []

    for idx in top_indices:
        results.append(
            (filtered_df.iloc[idx], similarity[idx])
        )

    return results


# ----------------------------
# SKILL MATCH
# ----------------------------
def skill_match(user_skills, tech_skills_text):

    user_skills = [
        s.strip().lower()
        for s in user_skills.split(",")
        if s.strip()
    ]

    tech_skills = re.split(r"[,\s]+", tech_skills_text.lower())
    tech_skills = list(dict.fromkeys(tech_skills))

    matched = set(user_skills).intersection(set(tech_skills))

    percent = 0

    if tech_skills:
        percent = int((len(matched) / len(tech_skills)) * 100)

    return tech_skills[:10], percent, matched


# ----------------------------
# CUSTOM DESIGN (FINAL)
# ----------------------------
st.markdown("""
<style>


/* =========================
   BACKGROUND
========================= */
.stApp {
    background: linear-gradient(135deg,#0F2027,#203A43,#2C5364);
}

section[data-testid="stSidebar"] {
    background: transparent !important;
}

section.main > div {
    background: transparent !important;
}

.block-container {
    background: transparent !important;
}




/* =========================
   HEADINGS
========================= */
h1, h2, h3 {
    color: #00F5D4 !important;
    text-align: left;
}


/* =========================
   MAIN REPORT CONTAINER
========================= */
.main-card {
    width: 100%;
    margin-top: 30px;
    background: rgba(255,255,255,0.08);
    padding: 35px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
}


/* =========================
   RESULT CARDS
========================= */
.result-card {
    width: 100%;
    background: rgba(255,255,255,0.06);
    padding: 25px;
    border-radius: 15px;
    margin: 25px 0;
}


/* =========================
   BUTTONS
========================= */
.stButton > button {
    background: linear-gradient(45deg,#00C9FF,#92FE9D) !important;
    color: black !important;
    border: none;
    padding: 10px 30px;
    border-radius: 25px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.05);
}


/* =========================
   INPUT FIELDS
========================= */
textarea, input {
    border-radius: 10px !important;
    background-color: black !important;
    color: white !important;
}




/* =========================
   LABEL SPACING
========================= */
label {
    margin-bottom: 6px !important;
}


/* =========================
   PROGRESS BAR
========================= */
.stProgress > div > div {
    background-color: #00F5D4 !important;
}

</style>
""", unsafe_allow_html=True)




# =====================================================
# STEP 1
# =====================================================
if st.session_state.step == 1:

    st.header("Step 1: Personal Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.selectbox(
            "Age Group",
            ["Below 15", "15-18", "18-22", "22-30", "30+"]
        )

    with col2:
        level = st.selectbox(
            "Education Level",
            [
                "10th", "11th", "12th", "Diploma",
                "UG", "PG", "Professional", "Housewife"
            ]
        )

    if st.button("Next ➡️"):
        st.session_state.age = age
        st.session_state.level = level
        st.session_state.step = 2
        st.rerun()


# =====================================================
# STEP 2
# =====================================================
elif st.session_state.step == 2:

    st.header("Step 2: Academic / Career Stream")

    stream = st.selectbox(
        "Select Your Stream",
        [
            "CSE", "ECE", "EEE", "CIVIL",
            "COMMERCE", "ARTS", "SCIENCE",
            "DESIGN", "ACCOUNTS", "MATHS"
        ]
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅️ Back"):
            st.session_state.step = 1
            st.rerun()

    with col2:
        if st.button("Next ➡️"):
            st.session_state.stream = stream
            st.session_state.step = 3
            st.rerun()


# =====================================================
# STEP 3
# =====================================================
elif st.session_state.step == 3:

    st.header("Step 3: Interests & Skills")

    user_input = st.text_area(
        "Describe Your Interests, Goals, and Passion"
    )

    user_skills = st.text_input(
        "Your Skills (comma separated)"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅️ Back"):
            st.session_state.step = 2
            st.rerun()

    with col2:
        if st.button("Predict Career 🎯"):

            if user_input.strip() == "":
                st.warning("Please describe your interests.")
            else:
                st.session_state.user_input = user_input
                st.session_state.user_skills = user_skills
                st.session_state.step = 4
                st.rerun()



# =====================================================
# STEP 4 : RESULTS
# =====================================================
# =====================================================
# STEP 4 : RESULTS
# =====================================================
elif st.session_state.step == 4:

    st.header("📊 Your Personalized Career Report")

    results = recommend(
        st.session_state.user_input,
        st.session_state.stream
    )

    if not results:
        st.error("No suitable career found. Try different inputs.")

        if st.button("🔁 Start Again"):
            st.session_state.step = 1
            st.rerun()

    else:

        for career, score in results:

            st.subheader(f"🎯 {career['Title']}")
            st.write(f"**Match Score:** {round(score * 100, 2)}%")

            # Career Overview
            st.subheader("📄 Career Overview")
            st.write(career["Description"])

            # Education Path
            st.subheader("🎓 Education Path")

            try:
                edu = int(career["Education"])

                if edu == 6:
                    st.write("Bachelor’s Degree Required")
                elif edu == 7:
                    st.write("Master’s Degree Required")
                else:
                    st.write("Advanced Degree Recommended")

            except:
                st.write("Refer to domain standards")

            # Skill Matching
            if st.session_state.user_skills.strip() != "":

                main_skills, match_percent, matched = skill_match(
                    st.session_state.user_skills,
                    career["Technology Skills"]
                )

                st.subheader("💻 Required Technical Skills")
                st.write(", ".join(main_skills))

                st.subheader("🧠 Skill Compatibility")
                st.progress(match_percent)
                st.write(f"{match_percent}% Match")

                if matched:
                    st.success("✅ Matching Skills")
                    st.write(", ".join(matched))

                missing = set(main_skills) - set(matched)

                if missing:
                    st.warning("❌ Skills to Improve")
                    st.write(", ".join(missing))

            # Career Roadmap
            st.subheader("🛣️ Career Roadmap")
            st.markdown("""
            - Foundation Learning  
            - Certification & Projects  
            - Internship / Practice  
            - Entry-Level Role  
            - Specialization  
            """)

            # Learning Resources
            st.subheader("📚 Learning Resources")
            st.markdown("""
            - Coursera  
            - NPTEL  
            - Udemy  
            - Kaggle  
            - YouTube Tech Channels  
            """)

            st.markdown("---")

        # Navigation Buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("⬅️ Go Back"):
                st.session_state.step = 3
                st.rerun()

        with col2:
            if st.button("🔁 Start Again"):
                st.session_state.step = 1
                st.rerun()


