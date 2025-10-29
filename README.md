#🧠 EduMatch – Course Recommendation System

🔗 Live Demo
[https://edumatch-heygahisfet9984pwtvbv4.streamlit.app/](https://edumatch-heygahisfet9984pwtvbv4.streamlit.app/)

# 📚 Context-Aware Course Recommendation System

This project is a hybrid recommendation system that suggests online courses from Coursera based on both content relevance and contextual user information such as time availability, device type, and study time preferences.

Built with:
- 🧠 Machine Learning (TF-IDF, cosine similarity)
- 📊 Streamlit for interactive UI
- 📁 Pandas & Scikit-learn for data processing and modeling

---

## 🔍 Features

- **Search by Skills or Keywords**  
  Input terms like `SQL`, `python`, or `data analysis` to find relevant courses.

- **Context-Aware Personalization**  
  Adjust recommendations by:
  - Available hours per week
  - Device used (desktop or mobile)
  - Preferred study time (morning/evening)

- **Flexible Filters**  
  Filter by course level, minimum rating, and top-N results.

- **Live Score Breakdown**  
  Each recommendation shows:
  - `score`: final ranking value  
  - `sim`: content similarity  
  - `pop`: popularity (rating + review count)  
  - `ctx`: context fit score

- **EDA Dashboard**  
  Visualize ratings, course levels, review counts, and most common skills.

---

## 🗃 Dataset

Uses a parsed and cleaned subset of Coursera course data including:
- `partner`, `course`, `skills`, `rating`, `reviewcount`, `level`, `certificatetype`, `duration`, and `crediteligibility`.

Derived columns:
- `reviewcount_num`, `skills_list`, `duration_months`, and a TF-IDF vector for text features.

---

   git clone https://github.com/your-username/context-aware-course-recommender.git
   cd context-aware-course-recommender
