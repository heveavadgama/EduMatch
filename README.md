# 🧠 EduMatch – Course Recommendation System using NLP and Transformers

### 🔗 Live Demo
[https://edumatch-heygahisfet9984pwtvbv4.streamlit.app/](https://edumatch-heygahisfet9984pwtvbv4.streamlit.app/)

---

## 📘 Project Overview
EduMatch is a **content-based course recommendation system** that uses **Natural Language Processing (NLP)** and **Transformer embeddings** to recommend courses based on similarity in textual content. It helps learners discover relevant online courses aligned with their interests.

The project compares **TF-IDF** and **Transformer-based (SentenceTransformer)** embeddings to analyze and recommend courses semantically.

---

## 🎯 Objective
- Build a **content-based recommender system** for course discovery.
- Implement and compare **TF-IDF** and **Transformer embeddings**.
- Visualize recommendations via **Streamlit web app**.
- Demonstrate understanding of **cosine similarity**, **feature extraction**, and **ranking** in recommendation systems.

---

## 🧩 Key Features
- 🧠 **Dual NLP Models:** TF-IDF + Transformer embeddings.
- ⚙️ **Content-Based Filtering:** Uses textual similarity.
- 🚀 **Instant Startup:** Precomputed embeddings for fast load.
- 🌐 **Streamlit Deployment:** Fully interactive and hosted online.
- 📊 **Explainable Recommendations:** Shows similarity scores.

---

## 📊 Dataset Details
- **File:** `coursea_data.csv`
- **Records:** 891
- **Attributes:** `course_title`, `course_organization`, `course_Certificate_type`, `course_rating`, `course_difficulty`, `course_students_enrolled`, `clean_text`
- **Precomputed Embeddings:** `course_embeddings.npy`

---

## 🧠 Techniques Used

| Technique | Description | Role |
|------------|--------------|------|
| **Content-Based Filtering** | Recommends items with similar content attributes. | Core recommendation logic |
| **TF-IDF Vectorization** | Captures importance of terms within text corpus. | Keyword-level similarity |
| **Transformer Embeddings** | Generates contextual semantic representations. | Deep semantic similarity |
| **Cosine Similarity** | Measures closeness between vectors. | Ranking and scoring |
| **Hybrid NLP Design** | Combines TF-IDF and Transformer. | Speed + accuracy trade-off |

---

## 🏗️ System Workflow
1. **Preprocessing:** Clean and normalize course text (lowercase, punctuation removal, etc.).
2. **Vectorization:** Convert text into numerical representations using TF-IDF and Transformers.
3. **Similarity Computation:** Compute cosine similarity between vectors.
4. **Recommendation Generation:** Retrieve top-K most similar courses.
5. **User Interaction:** Streamlit UI for example-course and free-text modes.

---

## 🧮 Mathematical Foundation
**Cosine Similarity:**

\$\$
\text{similarity}(A, B) = \frac{A \cdot B}{||A|| ||B||}
\$\$

A higher value indicates greater similarity between two courses.

---

## ⚙️ How to Run Locally

### Step 1: Clone the repository
```bash
git clone https://github.com/<yourusername>/EduMatch-Recommender.git
cd EduMatch-Recommender
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit App
```bash
streamlit run app.py
```

Then open: **http://localhost:8501**

---

## 🌐 Live App
✅ [EduMatch Live on Streamlit Cloud](https://edumatch-heygahisfet9984pwtvbv4.streamlit.app/)

---

## 📈 Sample Results

**Query:** “Beginner Python programming”

### Transformer Recommendations
| Course Title | Organization | Score |
|---------------|--------------|--------|
| Using Python to Access Web Data | University of Michigan | 0.204 |
| Introducción a la programación en Python I | Universidad Católica de Chile | 0.191 |

### TF-IDF Recommendations
| Course Title | Organization | Score |
|---------------|--------------|--------|
| Python Basics | University of Michigan | 0.499 |
| Data Analysis with Python | IBM | 0.474 |

---

## 🧩 Concepts from Recommendation Systems Covered
- ✅ Content-Based Filtering
- ✅ Cosine Similarity
- ✅ NLP-based Feature Extraction (TF-IDF, Transformer)
- ✅ Ranking and Top-K Retrieval
- ⚙️ Hybrid NLP Design (TF-IDF + Transformer)
- ⚙️ Explainability using similarity scores

---

## 🎓 Learning Outcomes
- Practical understanding of **content-based recommender systems**.
- Comparison of **classical vs deep NLP** representations.
- Implementation of **cosine similarity and feature engineering**.
- Experience with **Streamlit app deployment**.

---
