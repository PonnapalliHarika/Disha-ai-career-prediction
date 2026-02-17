import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Disha AI - Career Predictor",
    page_icon="🎯",
    layout="wide"
)

# ============================================
# CUSTOM CSS (Premium UI Styling)
# ============================================

st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
.big-title {
    font-size: 42px !important;
    font-weight: 800;
    color: #4B8BBE;
}
.subtitle {
    font-size: 18px !important;
    color: gray;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    background: linear-gradient(135deg, #e3f2fd, #ffffff);
    margin-bottom: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}
.footer {
    text-align:center;
    color:gray;
    font-size:14px;
    margin-top:40px;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================

st.markdown('<p class="big-title">🎯 Disha AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI Powered Inclusive Career Path Prediction System</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================
# LOAD MODEL & VECTORIZER
# ============================================

model = pickle.load(open("career_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ============================================
# SIDEBAR INPUT SECTION
# ============================================

st.sidebar.header("👤 Enter Your Details")

age = st.sidebar.number_input("Your Age", 14, 60)

education_level = st.sidebar.selectbox(
    "Education Level",
    ["10th","11th","12th","Diploma","Undergraduate","Postgraduate","Working Professional"]
)

stream = st.sidebar.selectbox(
    "Stream",
    ["Science","Commerce","Arts","Technical","Non-Technical"]
)

ug_course = st.sidebar.text_input("UG Course")
ug_specialization = st.sidebar.text_input("Specialization")
current_job = st.sidebar.text_input("Current Job")
years_experience = st.sidebar.number_input("Years of Experience", 0, 40)

interests = st.sidebar.text_area("Interests")
skills = st.sidebar.text_area("Skills")

preferred_work_type = st.sidebar.selectbox(
    "Preferred Work Type",
    ["Full-time","Remote","Government","Creative","Business"]
)

career_gap = st.sidebar.selectbox("Career Gap?", ["No","Yes"])

predict_button = st.sidebar.button("🚀 Predict Career")

# ============================================
# EDUCATION STAGE LOGIC
# ============================================

if education_level in ["10th", "11th", "12th"]:
    st.info("🎓 Early academic stage detected. Beginner-friendly career paths will be suggested.")

elif education_level == "Diploma":
    st.info("🛠 Diploma background detected. Skill-based and technical roles prioritized.")

elif education_level in ["Undergraduate", "Postgraduate"]:
    st.info("🎯 Specialized recommendations based on your education & skills.")

elif education_level == "Working Professional":
    st.info("💼 Career switch & upskilling pathways will be suggested.")

if career_gap == "Yes":
    st.warning("🔄 Career gap detected. Restart-friendly opportunities included.")

# ============================================
# CAREER ROADMAP FUNCTION
# ============================================

def get_roadmap(career):
    roadmaps = {
        "data analyst": [
            "Learn Python, SQL, Excel",
            "Understand statistics & data visualization",
            "Build 3–5 portfolio projects",
            "Apply for data analyst roles"
        ],
        "software developer": [
            "Master programming fundamentals",
            "Practice DSA regularly",
            "Build full-stack projects",
            "Apply for developer roles"
        ],
        "teacher": [
            "Strengthen subject knowledge",
            "Improve communication skills",
            "Gain certification if required",
            "Apply for teaching positions"
        ]
    }

    return roadmaps.get(
        career.lower(),
        ["Identify required skills", "Build relevant projects", "Apply for suitable roles"]
    )

# ============================================
# PREDICTION SECTION
# ============================================

if predict_button:

    if ug_course and ug_specialization and interests and skills:

        user_input = (
            "age_" + str(age) + " " +
            "edu_" + education_level + " " +
            "stream_" + stream + " " +
            "course_" + ug_course + " " +
            "spec_" + ug_specialization + " " +
            "job_" + current_job + " " +
            "exp_" + str(years_experience) + " " +
            "interest_" + interests + " " +
            "skill_" + skills + " " +
            "work_" + preferred_work_type + " " +
            "gap_" + career_gap
        )

        user_vector = vectorizer.transform([user_input])
        probabilities = model.predict_proba(user_vector)[0]
        careers = model.classes_

        results = sorted(
            zip(careers, probabilities),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        st.subheader("🔮 Top Career Recommendations")

        chart_data = []

        for career, prob in results:

            st.markdown(f"""
            <div class="result-box">
                <h4>{career.title()}</h4>
                <p>Confidence: <b>{round(prob*100,2)}%</b></p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(float(prob))

            chart_data.append((career.title(), prob))

        # ============================================
        # Probability Chart
        # ============================================

        st.subheader("📊 Prediction Confidence Chart")

        chart_df = pd.DataFrame(chart_data, columns=["Career", "Probability"])
        st.bar_chart(chart_df.set_index("Career"))

        # ============================================
        # Roadmaps
        # ============================================

        st.subheader("🛣 Suggested Career Roadmaps")

        for career, _ in results[:2]:
            st.markdown(f"### 🚀 {career.title()}")
            roadmap = get_roadmap(career)
            for step in roadmap:
                st.write("✅", step)

        st.success("Prediction Generated Successfully!")
        st.balloons()

    else:
        st.warning("⚠ Please fill required fields")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown('<div class="footer">© 2026 Disha AI | Built with ❤️ using Machine Learning & Streamlit</div>', unsafe_allow_html=True)

