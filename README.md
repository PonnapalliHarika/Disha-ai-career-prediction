🎯 Disha AI – Career Path Prediction Engine
📌 Overview
Disha AI is an intelligent Career Path Prediction Engine designed to help students and early‑career professionals make data‑driven career decisions.
The system analyzes a user’s educational background, skills, and interests and predicts the most suitable career paths along with probability scores and a structured roadmap to achieve the recommended career.

Traditional career counseling systems are often static, generic, and heavily dependent on grades. Disha AI overcomes these limitations by using Machine Learning, Natural Language Processing (NLP), and Data Analytics to provide personalized and adaptive career guidance.

❓ Problem Statement
Many students and early‑career professionals struggle to choose the right career path due to:

Lack of personalized guidance

Over‑reliance on academic scores

Limited awareness of emerging job roles

Static and generic career counseling systems

This often leads to:

Poor career decisions

Skill mismatch

Job dissatisfaction

Unemployment or underemployment

✅ Proposed Solution
Disha AI addresses these challenges by:

Analyzing individual skills, interests, and educational background

Predicting multiple career options with probabilities

Identifying the best‑fit career

Providing a clear step‑by‑step roadmap to achieve the recommended role

Dynamically adapting predictions based on updated inputs

🚀 Features
🎯 Personalized career recommendations

📊 Probability‑based career predictions (Top 3 roles)

🧠 Machine Learning + NLP powered analysis

🛣️ Career roadmap generation

🌐 Interactive web interface using Streamlit

☁️ Cloud deployment (Streamlit Community Cloud)

👥 Team collaboration using GitHub

🧠 System Architecture
User Input (Course, Specialization, Skills, Interests)

Text Preprocessing & Feature Engineering (TF‑IDF)

Machine Learning Model (Logistic Regression)

Career Prediction with Probabilities

Roadmap Generation

Streamlit UI Output

🛠️ Technologies Used
Programming Language: Python

Machine Learning: Scikit‑learn

NLP: TF‑IDF Vectorization

Data Handling: Pandas, NumPy

Web Framework: Streamlit

Version Control: Git & GitHub

Deployment: Streamlit Community Cloud

📂 Project Structure
Disha-AI/
│
├── app.py # Main Streamlit application
├── career_model.pkl # Trained ML model
├── vectorizer.pkl # Text vectorizer
├── career_cleaned.csv # Cleaned dataset
├── requirements.txt # Required libraries
│
├── step4_features.py # Feature engineering script
├── step5_train_model.py # Model training script
│
└── assets/ # Images / supporting files
📊 Machine Learning Approach
Input Features:

UG Course

Specialization

Skills

Interests

Feature Engineering:

Text features combined into a single column

TF‑IDF Vectorization

Model Used:

Logistic Regression

Chosen for simplicity, interpretability, and suitability for multi‑class classification

Output:

Top 3 predicted career roles

Probability score for each role

🛣️ Career Roadmap Generation
For the top predicted career, the system provides a structured roadmap such as:

Core skills to learn

Tools and technologies

Project building steps

Internship / experience guidance

Job application readiness

This roadmap helps users move from confusion to action.

🌐 Deployment
The application is deployed using Streamlit Community Cloud, making it publicly accessible without requiring a backend server.

Deployment Steps:

Push project to GitHub

Connect GitHub repository to Streamlit Cloud

Deploy app.py

Access via live URL

👥 Team Members
Harika Ponnapalli – Team Lead

Majji Poojitha – Team Member


🔮 Future Enhancements
Add personality assessment questionnaires

Use deep learning models (BERT / Transformers)

Integrate real‑time job market data

User login & profile tracking

Recommendation of courses and certifications

📜 Conclusion
Disha AI successfully demonstrates how Artificial Intelligence and Machine Learning can be used to solve real‑world problems in career guidance.
The system provides a scalable, intelligent, and user‑friendly solution to reduce career confusion and improve employability.

🌐 Live Application
👉 https://disha-ai-career-prediction-xxwiet8tmmyusgkjkbhytj.streamlit.app/
