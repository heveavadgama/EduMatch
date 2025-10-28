# app.py
"""
Course Recommender — use transformer for text queries (lazy-loaded)
Requires coursea_data.csv and course_embeddings.npy in repo root.

Behavior:
- Loads precomputed embeddings immediately for fast course->course recommendations.
- Loads SentenceTransformer model only when a text query is requested.
- TF-IDF fallback available as option in sidebar.
"""
import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# lazy import placeholder
SentenceTransformer = None

# ---------- Files ----------
CSV = "coursera_data.csv"
NPY = "course_embeddings.npy"
TRANSFORMER_NAME = "paraphrase-MiniLM-L3-v2"  # compact, good quality

# ---------- Helpers ----------
def clean_text_raw(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())

def top_k_from_sim(sim_vec, k=5, exclude_idx=None):
    sim = sim_vec.copy()
    if exclude_idx is not None:
        sim[exclude_idx] = -1.0
    idxs = np.argsort(-sim)[:k]
    return idxs, sim[idxs]

# ---------- Data and embeddings ----------
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(CSV)
    if "clean_text" not in df.columns:
        df["text_data"] = (
            df.get("course_title", "").fillna("") + " " +
            df.get("course_organization", "").fillna("") + " " +
            df.get("course_Certificate_type", "").fillna("") + " " +
            df.get("course_difficulty", "").fillna("")
        )
        df["clean_text"] = df["text_data"].apply(clean_text_raw)
    return df

@st.cache_resource(show_spinner=False)
def load_embeddings():
    emb = np.load(NPY)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_norm = emb / norms
    return emb_norm

# ---------- Transformer loader (lazy, cached) ----------
@st.cache_resource(show_spinner=False)
def load_transformer(model_name: str):
    global SentenceTransformer
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as ST
        SentenceTransformer = ST
    model = SentenceTransformer(model_name)
    return model

# ---------- TF-IDF builder ----------
@st.cache_resource(show_spinner=False)
def build_tfidf(df):
    tfidf = TfidfVectorizer(max_features=4000, stop_words="english")
    tfidf_mat = tfidf.fit_transform(df["clean_text"].astype(str))
    return tfidf, tfidf_mat

# ---------- Recommenders ----------
def recommend_by_index(idx, emb_norm, df, k=5):
    sim = emb_norm.dot(emb_norm[idx])
    idxs, scores = top_k_from_sim(sim, k=k, exclude_idx=idx)
    recs = []
    for j, i in enumerate(idxs):
        recs.append({
            "course_title": str(df.loc[i, "course_title"]),
            "org": str(df.loc[i, "course_organization"]),
            "rating": float(df.loc[i, "course_rating"]) if "course_rating" in df.columns else np.nan,
            "score": float(scores[j]),
        })
    return recs

def recommend_by_text_transformer(query, model, emb_norm, df, k=5):
    q = clean_text_raw(query)
    q_emb = model.encode([q])
    q_emb_n = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    sim = emb_norm.dot(q_emb_n[0])
    idxs, scores = top_k_from_sim(sim, k=k, exclude_idx=None)
    recs = []
    for j, i in enumerate(idxs):
        recs.append({
            "course_title": str(df.loc[i, "course_title"]),
            "org": str(df.loc[i, "course_organization"]),
            "rating": float(df.loc[i, "course_rating"]) if "course_rating" in df.columns else np.nan,
            "score": float(scores[j]),
        })
    return recs

def recommend_by_text_tfidf(query, tfidf_mat, tfidf, df, k=5):
    q = clean_text_raw(query)
    v = tfidf.transform([q])
    sims = cosine_similarity(v, tfidf_mat).ravel()
    idxs = sims.argsort()[::-1][:k]
    recs = []
    for i in idxs:
        recs.append({
            "course_title": str(df.loc[i, "course_title"]),
            "org": str(df.loc[i, "course_organization"]),
            "rating": float(df.loc[i, "course_rating"]) if "course_rating" in df.columns else np.nan,
            "score": float(sims[i]),
        })
    return recs

# ---------- App UI ----------
st.set_page_config(page_title="Course Recommender (Transformer Queries)", layout="wide")
st.title("Course Recommender — transformer for text queries")

df = load_data()
emb_norm = load_embeddings()
tfidf, tfidf_mat = build_tfidf(df)  # TF-IDF ready as fallback or hybrid

st.sidebar.header("Settings")
mode = st.sidebar.radio("Mode", ["By example course", "By text query"])
k = st.sidebar.slider("Top K", 1, 10, 5)
query_encoder_choice = st.sidebar.radio("Query encoder", ["Transformer (better quality)", "TF-IDF (faster)"], index=0)

if mode == "By example course":
    st.sidebar.markdown("Select a course to find similar courses.")
    sel = st.selectbox("Choose a course", ["-- pick --"] + df["course_title"].astype(str).tolist())
    if sel != "-- pick --":
        idx = int(df.index[df["course_title"].astype(str) == sel][0])
        st.subheader("Selected course")
        st.write(df.loc[idx, ["course_title", "course_organization", "course_rating"]])
        with st.spinner("Computing recommendations..."):
            recs = recommend_by_index(idx, emb_norm, df, k=k)
        st.subheader("Recommended courses")
        for r in recs:
            st.markdown(f"**{r['course_title']}**  \nOrg: {r['org']}  \nRating: {r['rating']}  \nSimilarity: {r['score']:.3f}")
            st.write("---")

else:
    st.sidebar.markdown("Enter a short text describing what you want to learn.")
    query = st.text_area("Query", placeholder="e.g., beginner python data analysis", height=120)
    if st.button("Recommend") and query.strip():
        if query_encoder_choice.startswith("Transformer"):
            # lazy-load transformer model (this may download on first run)
            with st.spinner("Loading transformer model (first time may take ~30-90s)..."):
                try:
                    model = load_transformer(TRANSFORMER_NAME)
                except Exception as e:
                    st.error(f"Failed to load transformer: {e}")
                    st.stop()
            with st.spinner("Encoding query and retrieving recommendations..."):
                recs = recommend_by_text_transformer(query, model, emb_norm, df, k=k)
        else:
            with st.spinner("Using TF-IDF for fast retrieval..."):
                recs = recommend_by_text_tfidf(query, tfidf_mat, tfidf, df, k=k)

        st.subheader("Recommendations")
        for r in recs:
            st.markdown(f"**{r['course_title']}**  \nOrg: {r['org']}  \nRating: {r['rating']}  \nScore: {r['score']:.3f}")
            st.write("---")

st.sidebar.markdown("### Notes")
st.sidebar.write("- Transformer queries give better semantic matches but the model downloads on first use.")
st.sidebar.write("- Precomputed embeddings are used for course->course similarity to keep that flow instant.")
