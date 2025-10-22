# app.py
"""
Faster-start Course Recommender (single-file)
Place coursea_data.csv in repo root. Optional: commit course_embeddings.npy for instant startup.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.preprocessing import normalize
import os

# try lazy import to show clearer errors
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# ---------- Config ----------
DATA_FILE = "coursera_data.csv"
EMBED_FILE = "course_embeddings.npy"   # recommended to commit to repo for instant start
EMBED_MODEL_NAME = "paraphrase-MiniLM-L3-v2"  # smaller & faster than all-MiniLM-L6-v2
MAX_ROWS_FOR_QUICK_START = 500  # set None to use full dataset

# ---------- Minimal stopwords ----------
_SIMPLE_STOPWORDS = {
    "the", "and", "a", "an", "of", "in", "for", "to", "with", "on", "by", "from",
    "is", "are", "course", "specialization", "introduction", "intro",
    "beginner", "intermediate", "advanced"
}

# ---------- Helpers ----------
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
        m = re.search(r'[\d\.]+', s)
        return float(m.group()) if m else np.nan

def clean_text_raw(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in _SIMPLE_STOPWORDS]
    return " ".join(tokens)

# ---------- Data loader ----------
@st.cache_data(show_spinner=False)
def load_dataframe(path: str, max_rows=None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found at {p.resolve()}. Add {p.name} to repo root.")
    df = pd.read_csv(p)
    if max_rows:
        df = df.head(max_rows).copy()
    df.columns = [c.strip() for c in df.columns]
    df['students_enrolled_num'] = df.get('course_students_enrolled', pd.Series([np.nan]*len(df))).apply(parse_enrollment)
    df['course_rating'] = pd.to_numeric(df.get('course_rating', pd.Series([np.nan]*len(df))), errors='coerce')
    df['course_difficulty'] = df.get('course_difficulty','').astype(str).replace('nan','')
    df['text_data'] = (
        df.get('course_title','').fillna('') + ' ' +
        df.get('course_organization','').fillna('') + ' ' +
        df.get('course_Certificate_type','').fillna('') + ' ' +
        df['course_difficulty'].fillna('')
    )
    df['clean_text'] = df['text_data'].apply(clean_text_raw)
    df = df.reset_index(drop=True)
    return df

# ---------- Embedding loader / builder ----------
@st.cache_resource(show_spinner=False)
def get_encoder_and_embeddings(df: pd.DataFrame, embed_file: str, model_name: str):
    # Try load precomputed embeddings
    if os.path.exists(embed_file):
        emb = np.load(embed_file)
        # ensure shape matches df length; if mismatch, ignore file
        if emb.shape[0] == len(df):
            emb_norm = normalize(np.array(emb), axis=1, norm='l2')
            return None, emb_norm  # model is None because we only load embeddings
        # else fallthrough to recompute
    # Compute embeddings (download model once)
    global SentenceTransformer
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as ST
        SentenceTransformer = ST
    model = SentenceTransformer(model_name)
    texts = df['clean_text'].astype(str).tolist()
    emb = model.encode(texts, show_progress_bar=True)
    emb_norm = normalize(np.array(emb), axis=1, norm='l2')
    # Save to disk for future fast startups
    try:
        np.save(embed_file, emb_norm)
    except Exception:
        pass
    return model, emb_norm

# ---------- Recommendation utils ----------
def top_k_from_sim(sim_vec: np.ndarray, k: int = 5, exclude_idx=None):
    sim = sim_vec.copy()
    if exclude_idx is not None:
        sim[exclude_idx] = -1.0
    idxs = np.argsort(-sim)[:k]
    return idxs, sim[idxs]

def recommend_by_index(idx: int, emb_norm: np.ndarray, df: pd.DataFrame, k: int = 5):
    sim = emb_norm.dot(emb_norm[idx])
    idxs, scores = top_k_from_sim(sim, k=k, exclude_idx=idx)
    return format_results(idxs, scores, df)

def recommend_by_text(query: str, model, emb_norm: np.ndarray, df: pd.DataFrame, k: int = 5):
    q_clean = clean_text_raw(query)
    # if model is None then embeddings were preloaded; load small model just for query encoding
    if model is None:
        global SentenceTransformer
        if SentenceTransformer is None:
            from sentence_transformers import SentenceTransformer as ST
            SentenceTransformer = ST
        model = SentenceTransformer(EMBED_MODEL_NAME)
    q_emb = model.encode([q_clean])
    q_emb_n = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    sim = emb_norm.dot(q_emb_n[0])
    idxs, scores = top_k_from_sim(sim, k=k, exclude_idx=None)
    return format_results(idxs, scores, df)

def format_results(idxs, scores, df):
    out = []
    for i, s in zip(idxs, scores):
        row = df.iloc[int(i)]
        out.append({
            "index": int(i),
            "course_title": str(row.get('course_title', '')),
            "organization": str(row.get('course_organization', '')),
            "rating": float(row.get('course_rating', np.nan)) if 'course_rating' in row else np.nan,
            "difficulty": str(row.get('course_difficulty', '')) if 'course_difficulty' in row else '',
            "score": float(s)
        })
    return out

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Course Recommender (fast start)", layout="wide")
st.title("Content-based Course Recommender — Faster Startup")

st.markdown(
    "This app prefers a precomputed `course_embeddings.npy` in the repo root. "
    "If absent it computes a smaller model embeddings (`paraphrase-MiniLM-L3-v2`) and saves them."
)

# Controls
use_sample = st.sidebar.checkbox(f"Limit to first {MAX_ROWS_FOR_QUICK_START} rows for faster start", value=True)
compute_now = st.sidebar.button("Force (re)compute embeddings now")
k = st.sidebar.slider("Top K", 1, 10, 5)
mode = st.sidebar.radio("Mode", ["By example course", "By text query"])

# Load data
max_rows = MAX_ROWS_FOR_QUICK_START if use_sample else None
with st.spinner("Loading dataset..."):
    try:
        df = load_dataframe(DATA_FILE, max_rows=max_rows)
        st.success(f"Loaded {len(df)} courses.")
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()

# Load or compute embeddings
with st.spinner("Loading/creating embeddings (fast path)..."):
    try:
        model, emb_norm = get_encoder_and_embeddings(df, EMBED_FILE, EMBED_MODEL_NAME)
        if os.path.exists(EMBED_FILE) and model is None:
            st.info("Loaded precomputed embeddings from disk. Model not loaded.")
        else:
            st.info("Computed embeddings using smaller model and saved to disk for future startups.")
    except Exception as e:
        st.error(f"Embedding step failed: {e}")
        st.stop()

# Option to force recompute (overwrite)
if compute_now:
    with st.spinner("Forcing recompute of embeddings..."):
        try:
            # remove existing file then recompute
            if os.path.exists(EMBED_FILE):
                os.remove(EMBED_FILE)
            model, emb_norm = get_encoder_and_embeddings(df, EMBED_FILE, EMBED_MODEL_NAME)
            st.success("Recomputed and saved embeddings.")
        except Exception as e:
            st.error(f"Failed to recompute: {e}")
            st.stop()

# UI actions
if mode == "By example course":
    st.sidebar.markdown("Pick a course example")
    course_list = df['course_title'].astype(str).tolist()
    sel = st.sidebar.selectbox("Select course", options=["-- pick --"] + course_list)
    if sel != "-- pick --":
        idx = df.index[df['course_title'].astype(str) == sel][0]
        st.subheader("Selected course")
        st.write(df.loc[idx, ['course_title', 'course_organization', 'course_rating', 'course_difficulty']])
        with st.spinner("Computing recommendations..."):
            recs = recommend_by_index(int(idx), emb_norm, df, k=k)
        st.subheader("Top recommendations")
        for r in recs:
            st.markdown(f"**{r['course_title']}**  \nOrg: {r['organization']}  \nRating: {r['rating']}  \nDifficulty: {r['difficulty']}  \nScore: {r['score']:.3f}")
            st.write("---")

else:
    st.sidebar.markdown("Type a short query describing what you want to learn.")
    query = st.sidebar.text_area("Query", height=120, placeholder="e.g., beginner python for data analysis")
    if st.sidebar.button("Recommend from query") and query.strip():
        with st.spinner("Encoding query and retrieving..."):
            recs = recommend_by_text(query, model, emb_norm, df, k=k)
        st.subheader("Top recommendations for your query")
        for r in recs:
            st.markdown(f"**{r['course_title']}**  \nOrg: {r['organization']}  \nRating: {r['rating']}  \nDifficulty: {r['difficulty']}  \nScore: {r['score']:.3f}")
            st.write("---")

st.sidebar.markdown("## Deployment notes")
st.sidebar.write("- For instant startup commit `course_embeddings.npy` to repo.")
st.sidebar.write("- Smaller embedding model used for faster load.")
st.sidebar.write("- Sample dataset enabled by default for quicker demos.")
