# app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.preprocessing import normalize

# Import SentenceTransformer lazily to avoid startup errors in environments without torch yet.
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # will import later when needed

# ---------- Config ----------
DATA_FILE = "coursera_data.csv"   # expected in repo root
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------- Minimal stopwords for speed ----------
_SIMPLE_STOPWORDS = {
    "the", "and", "a", "an", "of", "in", "for", "to", "with", "on", "by", "from",
    "is", "are", "course", "specialization", "introduction", "intro",
    "beginner", "intermediate", "advanced"
}

# ---------- Utility functions ----------
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

# ---------- Data loading / preprocessing ----------
@st.cache_data(show_spinner=False)
def load_dataframe(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found at {p.resolve()}. Add {p.name} to repo root.")
    df = pd.read_csv(p)
    # standardize column names
    df.columns = [c.strip() for c in df.columns]
    # parse enrollment and rating safely
    if 'course_students_enrolled' in df.columns:
        df['students_enrolled_num'] = df['course_students_enrolled'].apply(parse_enrollment)
    else:
        df['students_enrolled_num'] = np.nan
    if 'course_rating' in df.columns:
        df['course_rating'] = pd.to_numeric(df['course_rating'], errors='coerce')
    else:
        df['course_rating'] = np.nan
    # ensure course_difficulty is string
    if 'course_difficulty' in df.columns:
        df['course_difficulty'] = df['course_difficulty'].astype(str).replace('nan', '')
    else:
        df['course_difficulty'] = ''
    # combine textual fields into text_data
    df['text_data'] = (
        df.get('course_title', '').fillna('') + ' ' +
        df.get('course_organization', '').fillna('') + ' ' +
        df.get('course_Certificate_type', '').fillna('') + ' ' +
        df['course_difficulty'].fillna('')
    )
    df['clean_text'] = df['text_data'].apply(clean_text_raw)
    df = df.reset_index(drop=True)
    return df

# ---------- Embedding model and embeddings ----------
@st.cache_resource(show_spinner=False)
def load_encoder_and_embeddings(df: pd.DataFrame):
    global SentenceTransformer
    if SentenceTransformer is None:
        # import here to surface clearer error messages if missing
        from sentence_transformers import SentenceTransformer as ST
        SentenceTransformer = ST
    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = df['clean_text'].astype(str).tolist()
    emb = model.encode(texts, show_progress_bar=False)
    emb_norm = normalize(np.array(emb), axis=1, norm='l2')
    return model, emb_norm

# ---------- Recommendation utilities ----------
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
st.set_page_config(page_title="Course Recommender", layout="wide")
st.title("Content-based Course Recommender (single-file)")

st.markdown(
    "This app builds sentence-transformer embeddings from course text and "
    "returns nearest courses by cosine similarity. Put `coursea_data.csv` in the repo root."
)

# Load dataframe
with st.spinner("Loading dataset..."):
    try:
        df = load_dataframe(DATA_FILE)
        st.success(f"Loaded {len(df)} courses.")
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()

# Optional sample and stats
if st.checkbox("Show sample data and stats"):
    cols = ['course_title', 'course_organization', 'course_rating', 'course_difficulty', 'students_enrolled_num']
    available = [c for c in cols if c in df.columns]
    st.dataframe(df[available].head(10))
    st.write("Rating summary")
    st.write(df['course_rating'].describe())

# Compute embeddings button (to avoid auto-running heavy download)
if st.button("Compute embeddings now (model download may occur once)"):
    with st.spinner("Loading embedding model and computing embeddings..."):
        try:
            model, emb_norm = load_encoder_and_embeddings(df)
            st.success("Embeddings computed.")
        except Exception as e:
            st.error(f"Failed to load model or compute embeddings: {e}")
            st.stop()
else:
    model = None
    emb_norm = None

# Sidebar controls
mode = st.sidebar.radio("Recommendation mode", ["By example course", "By text query"])
k = st.sidebar.slider("Number of recommendations (k)", 1, 10, 5)

if mode == "By example course":
    st.sidebar.markdown("Pick one example course to find similar courses.")
    course_list = df['course_title'].astype(str).tolist()
    selection = st.sidebar.selectbox("Select course", options=["-- pick --"] + course_list)
    if selection != "-- pick --":
        idx = df.index[df['course_title'].astype(str) == selection][0]
        st.subheader("Selected course")
        st.write(df.loc[idx, ['course_title', 'course_organization', 'course_rating', 'course_difficulty']])
        if model is None or emb_norm is None:
            with st.spinner("Loading model and computing embeddings..."):
                try:
                    model, emb_norm = load_encoder_and_embeddings(df)
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
                    st.stop()
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
        if model is None or emb_norm is None:
            with st.spinner("Loading model and computing embeddings..."):
                try:
                    model, emb_norm = load_encoder_and_embeddings(df)
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
                    st.stop()
        with st.spinner("Encoding query and retrieving recommendations..."):
            recs = recommend_by_text(query, model, emb_norm, df, k=k)
        st.subheader("Top recommendations for your query")
        for r in recs:
            st.markdown(f"**{r['course_title']}**  \nOrg: {r['organization']}  \nRating: {r['rating']}  \nDifficulty: {r['difficulty']}  \nScore: {r['score']:.3f}")
            st.write("---")

st.sidebar.markdown("## Notes")
st.sidebar.write("- This single-file app computes embeddings at runtime and does not persist model files.")
st.sidebar.write(f"- Embedding model: {EMBED_MODEL_NAME}")
st.sidebar.write("- If deploying on Streamlit Cloud or similar, ensure the instance has enough memory. Consider using a sampled CSV for faster startup.")
