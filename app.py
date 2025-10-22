# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# ---------- Config ----------
DATA_PATH = "/content/drive/MyDrive/NLP_RS/extracted/coursera_data.csv"  # update if needed
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------- Utilities ----------
def parse_enrollment(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower().replace(',', '')
    try:
        if s.endswith('k'):
            return float(s[:-1]) * 1_000
        if s.endswith('m'):
            return float(s[:-1]) * 1_000_000
        return float(s)
    except:
        import re
        m = re.search(r'[\d\.]+', s)
        return float(m.group()) if m else np.nan

_SIMPLE_STOPWORDS = {
    "the","and","a","an","of","in","for","to","with","on","by","from","is","are",
    "course","specialization","introduction","intro","beginner","intermediate","advanced"
}

def clean_text_raw(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in _SIMPLE_STOPWORDS]
    return " ".join(tokens)

# ---------- Cached loaders ----------
@st.cache_resource(show_spinner=False)
def load_dataframe(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found at {path}")
    df_local = pd.read_csv(p)
    # basic cleaning and type fixes
    df_local = df_local.rename(columns=lambda c: c.strip())
    if 'course_students_enrolled' in df_local.columns:
        df_local['students_enrolled_num'] = df_local['course_students_enrolled'].apply(parse_enrollment)
    else:
        df_local['students_enrolled_num'] = np.nan
    if 'course_rating' in df_local.columns:
        df_local['course_rating'] = pd.to_numeric(df_local['course_rating'], errors='coerce')
    else:
        df_local['course_rating'] = np.nan
    # ensure difficulty is string
    if 'course_difficulty' in df_local.columns:
        df_local['course_difficulty'] = df_local['course_difficulty'].astype(str).replace('nan','')
    else:
        df_local['course_difficulty'] = ''
    # create text representation
    df_local['text_data'] = (
        df_local.get('course_title','').fillna('') + ' ' +
        df_local.get('course_organization','').fillna('') + ' ' +
        df_local.get('course_Certificate_type','').fillna('') + ' ' +
        df_local['course_difficulty'].fillna('')
    )
    df_local['clean_text'] = df_local['text_data'].apply(clean_text_raw)
    df_local = df_local.reset_index(drop=True)
    return df_local

@st.cache_resource(show_spinner=False)
def load_encoder_and_embeddings(df):
    # load embedding model
    model = SentenceTransformer(EMBED_MODEL_NAME)
    # compute embeddings for all courses
    texts = df['clean_text'].tolist()
    emb = model.encode(texts, show_progress_bar=False)
    emb_norm = normalize(np.array(emb), axis=1, norm='l2')
    return model, emb_norm

# ---------- Recommendation logic ----------
def top_k_from_sim(sim_vec, k=5, exclude_idx=None):
    sim = sim_vec.copy()
    if exclude_idx is not None:
        sim[exclude_idx] = -1.0
    idxs = np.argsort(-sim)[:k]
    return idxs, sim[idxs]

def recommend_by_index(idx, emb_norm, df, k=5):
    sim = emb_norm.dot(emb_norm[idx])
    idxs, scores = top_k_from_sim(sim, k=k, exclude_idx=idx)
    return format_results(idxs, scores, df)

def recommend_by_text(query, model, emb_norm, df, k=5):
    q_clean = clean_text_raw(query)
    q_emb = model.encode([q_clean])
    q_emb_n = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    sim = emb_norm.dot(q_emb_n[0])
    idxs, scores = top_k_from_sim(sim, k=k, exclude_idx=None)
    return format_results(idxs, scores, df)

def format_results(idxs, scores, df):
    out = []
    for i, s in zip(idxs, scores):
        r = {
            "index": int(i),
            "course_title": str(df.loc[i, 'course_title']) if 'course_title' in df.columns else '',
            "organization": str(df.loc[i, 'course_organization']) if 'course_organization' in df.columns else '',
            "rating": float(df.loc[i, 'course_rating']) if 'course_rating' in df.columns else np.nan,
            "difficulty": str(df.loc[i, 'course_difficulty']) if 'course_difficulty' in df.columns else '',
            "score": float(s)
        }
        out.append(r)
    return out

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Content-based Course Recommender (All-in-one)", layout="wide")
st.title("Content-based Course Recommender — All logic in one file")

st.markdown("Load dataset, compute embeddings, and get recommendations using cosine similarity on sentence-transformer embeddings.")

# Load data
with st.spinner("Loading data..."):
    try:
        df = load_dataframe(DATA_PATH)
        st.success(f"Loaded {len(df)} courses from CSV.")
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        st.stop()

# Show small EDA
if st.checkbox("Show sample data and basic stats"):
    st.dataframe(df[['course_title','course_organization','course_rating','course_difficulty','students_enrolled_num']].head(10))
    st.write("Ratings summary:")
    st.write(df['course_rating'].describe())

# Load encoder and embeddings
if st.button("Compute embeddings now (may take ~10s)"):
    with st.spinner("Loading model and computing embeddings..."):
        model, emb_norm = load_encoder_and_embeddings(df)
    st.success("Embeddings computed and normalized.")
else:
    # lazy load only when needed
    model = None
    emb_norm = None

# Recommendation mode
mode = st.sidebar.radio("Mode", ["By example course", "By text query"])
k = st.sidebar.slider("Top K", 1, 10, 5)

if mode == "By example course":
    st.sidebar.markdown("Select a course to find similar courses.")
    course_list = df['course_title'].astype(str).tolist()
    sel = st.sidebar.selectbox("Course", options=["-- pick --"] + course_list)
    if sel != "-- pick --":
        idx = df.index[df['course_title'].astype(str) == sel][0]
        st.subheader("Selected course")
        st.write(df.loc[idx, ['course_title','course_organization','course_rating','course_difficulty']])
        if model is None or emb_norm is None:
            with st.spinner("Loading encoder and computing embeddings..."):
                model, emb_norm = load_encoder_and_embeddings(df)
        with st.spinner("Computing recommendations..."):
            recs = recommend_by_index(int(idx), emb_norm, df, k=k)
        st.subheader("Top recommendations")
        for r in recs:
            st.markdown(f"**{r['course_title']}**  \nOrg: {r['organization']}  \nRating: {r['rating']}  \nDifficulty: {r['difficulty']}  \nScore: {r['score']:.3f}")
            st.write("---")

else:
    st.sidebar.markdown("Enter a short text describing what you want to learn.")
    query = st.sidebar.text_area("Query", height=120)
    if st.sidebar.button("Recommend from text") and query.strip():
        if model is None or emb_norm is None:
            with st.spinner("Loading encoder and computing embeddings..."):
                model, emb_norm = load_encoder_and_embeddings(df)
        with st.spinner("Encoding query and retrieving..."):
            recs = recommend_by_text(query, model, emb_norm, df, k=k)
        st.subheader("Top recommendations for your query")
        for r in recs:
            st.markdown(f"**{r['course_title']}**  \nOrg: {r['organization']}  \nRating: {r['rating']}  \nDifficulty: {r['difficulty']}  \nScore: {r['score']:.3f}")
            st.write("---")

st.sidebar.markdown("### Notes")
st.sidebar.write("- This app does not persist models or embeddings to disk. All logic runs in this file.")
st.sidebar.write("- Model loaded: " + EMBED_MODEL_NAME)
st.sidebar.write("- For faster demo, pick small k and avoid recomputing embeddings repeatedly.")
