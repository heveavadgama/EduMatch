# Cell 8 — write streamlit app file
app_code = r'''
import streamlit as st
import pandas as pd, numpy as np, re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "udemy_courses_cleaned.csv"
EMB_PATH = "udemy_course_embeddings.npy"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+',' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return ' '.join(text.split())

df = pd.read_csv(DATA_PATH)
if 'clean_text' not in df.columns:
    df['text_data'] = df['course_title'].fillna('') + ' ' + df['subject'].fillna('') + ' ' + df['level_simple'].fillna('')
    df['clean_text'] = df['text_data'].apply(clean_text)

tfidf = TfidfVectorizer(max_features=5000)
tfidf_mat = tfidf.fit_transform(df['clean_text'].astype(str))

emb_norm = None
if Path(EMB_PATH).exists():
    emb_norm = np.load(EMB_PATH)

st.title("Udemy Course Recommender (Context-Aware)")
skill = st.sidebar.selectbox("Skill level", ["Beginner","Intermediate","Advanced"])
hours = st.sidebar.slider("Hours per week available", 0.5, 40.0, 4.0, step=0.5)
device = st.sidebar.selectbox("Device", ["Desktop","Mobile"])
alpha = st.sidebar.slider("Weight on semantic similarity", 0.0, 1.0, 0.7, step=0.05)
use_transformer = st.sidebar.checkbox("Use Transformer encoding for query (slow first time)", value=False)

mode = st.radio("Mode", ["By query", "By example course"])
k = st.slider("Top K", 1, 15, 5)

def semantic_by_query_tf(query):
    q = clean_text(query)
    qv = tfidf.transform([q])
    sims = cosine_similarity(qv, tfidf_mat).ravel()
    mn, mx = sims.min(), sims.max()
    return (sims - mn) / (mx - mn + 1e-9)

def semantic_by_query_transformer(query, model):
    q = clean_text(query)
    q_emb = model.encode([q])
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    sims = emb_norm.dot(q_emb[0])
    return np.clip(sims, 0, 1)

def time_score_cont(user_hours, course_hours, expected_weeks=4):
    try:
        if np.isnan(course_hours) or course_hours<=0:
            return 0.5
        required_weekly = max(0.5, course_hours / expected_weeks)
        return float(max(0, min(1, user_hours / required_weekly)))
    except:
        return 0.5

def skill_score(user_skill, course_level):
    map_v = {'Beginner':0, 'All':0, 'Intermediate':1, 'Advanced':2, 'Unknown':1}
    us = {'Beginner':0,'Intermediate':1,'Advanced':2}.get(user_skill,1)
    cs = map_v.get(course_level,1)
    dist = abs(us - cs)
    if dist == 0: return 1.0
    if dist == 1: return 0.6
    return 0.2

if mode == "By query":
    query = st.text_area("Enter query", height=120)
    if st.button("Recommend") and query.strip():
        if use_transformer and emb_norm is not None:
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
                sem = semantic_by_query_transformer(query, model)
            except Exception as e:
                st.error("Transformer load failed: " + str(e))
                sem = semantic_by_query_tf(query)
        else:
            sem = semantic_by_query_tf(query)

        # compute context scores
        skill_scores = df['level_simple'].apply(lambda lvl: skill_score(skill, lvl)).values
        time_scores = df['content_duration_hr'].fillna(0).apply(lambda ch: time_score_cont(hours, ch)).values
        context_scores = 0.6*skill_scores + 0.4*time_scores
        final = alpha * sem + (1-alpha) * context_scores
        top_idxs = np.argsort(-final)[:k]
        for idx in top_idxs:
            st.markdown(f"**{df.loc[idx,'course_title']}**")
            st.write(f"Subject: {df.loc[idx,'subject']} | Duration (hrs): {df.loc[idx,'content_duration_hr']} | Level: {df.loc[idx,'level_simple']} | Score: {final[idx]:.3f}")
            st.write("---")

else:
    choice = st.selectbox("Pick a course", ["-- pick --"] + df['course_title'].tolist())
    if choice != "-- pick --":
        idx = df.index[df['course_title']==choice][0]
        # semantic similarity by embeddings if available else tfidf
        if emb_norm is not None:
            sem = emb_norm.dot(emb_norm[idx])
        else:
            v = tfidf_mat[idx]
            sem = cosine_similarity(v, tfidf_mat).ravel()
            mn, mx = sem.min(), sem.max()
            sem = (sem-mn)/(mx-mn+1e-9)
        skill_scores = df['level_simple'].apply(lambda lvl: skill_score(skill, lvl)).values
        time_scores = df['content_duration_hr'].fillna(0).apply(lambda ch: time_score_cont(hours, ch)).values
        context_scores = 0.6*skill_scores + 0.4*time_scores
        final = alpha * sem + (1-alpha) * context_scores
        final[idx] = -1
        top_idxs = np.argsort(-final)[:k]
        for i in top_idxs:
            st.markdown(f"**{df.loc[i,'course_title']}**")
            st.write(f"Subject: {df.loc[i,'subject']} | Duration (hrs): {df.loc[i,'content_duration_hr']} | Level: {df.loc[i,'level_simple']} | Score: {final[i]:.3f}")
            st.write("---")
'''
with open("udemy_recommender_app.py","w") as f:
    f.write(app_code)
print("Wrote udemy_recommender_app.py")
