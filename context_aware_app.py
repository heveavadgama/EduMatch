# context_aware_app.py
"""
Context-Aware Course Recommender (Streamlit single-file)
Place the following files in the same folder:
 - coursea_data.csv
 - course_embeddings.npy  (precomputed, normalized, float32 recommended)

Run:
    pip install -r requirements.txt
    streamlit run context_aware_app.py

Notes:
 - Uses embeddings for course->course similarity (fast if .npy present).
 - TF-IDF used by default for text queries for fast response.
 - Transformer query encoding is available as an option (lazy-loaded).
 - Context matching is rule-based and explainable; replace with learned model for production.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try to lazy-import SentenceTransformer only if needed
SentenceTransformer = None

# ---------- Config ----------
CSV = "coursera_data.csv"
NPY = "course_embeddings.npy"
TRANSFORMER_MODEL = "paraphrase-MiniLM-L3-v2"  # small & decent
DEFAULT_TOPK = 5

# ---------- Utilities ----------
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

# ---------- Load data and embeddings ----------
@st.cache_data(show_spinner=False)
def load_df(path=CSV):
    p = Path(path)
    if not p.exists():
        st.error(f"CSV not found at {p.resolve()}. Add {p.name} to repo root.")
        st.stop()
    df = pd.read_csv(p)
    # create clean_text if not present
    if "clean_text" not in df.columns:
        df["text_data"] = (
            df.get("course_title", "").fillna("") + " " +
            df.get("course_organization", "").fillna("") + " " +
            df.get("course_Certificate_type", "").fillna("") + " " +
            df.get("course_difficulty", "").fillna("")
        )
        df["clean_text"] = df["text_data"].apply(clean_text_raw)
    # ensure expected columns exist to avoid KeyErrors later
    for col in ["course_title","course_organization","course_Certificate_type","course_rating","course_difficulty"]:
        if col not in df.columns:
            df[col] = ""
    return df.reset_index(drop=True)

@st.cache_resource(show_spinner=False)
def load_embeddings(path=NPY):
    p = Path(path)
    if not p.exists():
        st.error(f"Embeddings file {path} not found. Generate and place course_embeddings.npy in repo root.")
        st.stop()
    emb = np.load(p)
    # ensure normalized
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    emb_norm = emb / norms
    return emb_norm

@st.cache_resource(show_spinner=False)
def build_tfidf(df):
    tfidf = TfidfVectorizer(max_features=4000, stop_words="english")
    mat = tfidf.fit_transform(df["clean_text"].astype(str))
    return tfidf, mat

@st.cache_resource(show_spinner=False)
def load_transformer_model(model_name=TRANSFORMER_MODEL):
    global SentenceTransformer
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as ST
        SentenceTransformer = ST
    return SentenceTransformer(model_name)

# ---------- Context modelling (rule-based, explainable) ----------
def context_score_for_course(user_ctx, course_row):
    """
    Compute a context compatibility score in [0,1] between a user context dict and one course row.
    This is rule-based and meant to be interpretable. Replace with learned model if you have logs.
    user_ctx keys:
      - skill_level: 'Beginner'|'Intermediate'|'Advanced'
      - hours_per_week: float
      - device: 'Mobile'|'Desktop'
      - preferred_time: 'Morning'|'Afternoon'|'Evening'
    course_row holds course fields like course_difficulty and course_Certificate_type.
    """
    score_components = []

    # 1) Skill-level match (primary)
    user_skill = user_ctx.get("skill_level","").lower()
    course_diff = str(course_row.get("course_difficulty","")).lower()
    # Rules:
    # - exact match: 1.0
    # - course marked 'mixed' -> 0.9
    # - user beginner and course beginner -> 1.0, if mismatch but course mixed -> 0.9
    # - adjacent levels (Beginner<->Intermediate) -> 0.6
    # - mismatch (Beginner vs Advanced) -> 0.1
    if "mixed" in course_diff:
        skill_score = 0.9
    elif user_skill and user_skill in course_diff:
        skill_score = 1.0
    else:
        # handle adjacency
        mapping = {"beginner":0, "intermediate":1, "advanced":2}
        try:
            us = mapping.get(user_skill, 1)
            cs = mapping.get(course_diff, 1)
            dist = abs(us - cs)
            if dist == 0:
                skill_score = 1.0
            elif dist == 1:
                skill_score = 0.6
            else:
                skill_score = 0.1
        except:
            skill_score = 0.6
    score_components.append(("skill", skill_score))

    # 2) Time availability vs course length proxy (secondary)
    # We don't have explicit course duration. Use Certificate type: SPECIALIZATION ~ long, COURSE ~ medium/short.
    user_hours = float(user_ctx.get("hours_per_week", 3.0))
    cert = str(course_row.get("course_Certificate_type","")).lower()
    if "specialization" in cert:
        course_length = "long"
    else:
        course_length = "short"  # coarse proxy
    # rule: if user hours < 3 and course 'short' -> good; if user hours low and course long -> bad
    if user_hours >= 8:
        time_score = 1.0
    elif user_hours >= 3:
        time_score = 0.9 if course_length == "short" else 0.8
    else:
        # very limited time
        time_score = 0.9 if course_length == "short" else 0.2
    score_components.append(("time", time_score))

    # 3) Device compatibility (weak)
    # If course_title mentions 'mobile' or 'app' treat as mobile-friendly. Otherwise neutral.
    device = user_ctx.get("device","Desktop").lower()
    title = str(course_row.get("course_title","")).lower()
    mobile_friendly = 1.0 if ("mobile" in title or "app" in title or "responsive" in title) else 0.8
    # if user uses mobile prefer mobile_friendly higher
    if device == "mobile":
        device_score = mobile_friendly
    else:
        device_score = 1.0  # desktop assumed compatible with most
    score_components.append(("device", device_score))

    # 4) Preferred study time (not in data). keep neutral unless we had session timestamps.
    pref_time_score = 1.0
    score_components.append(("preferred_time", pref_time_score))

    # Combine components as weighted mean (weights can be tuned)
    # By default put more weight on skill and time
    weights = {
        "skill": 0.45,
        "time": 0.30,
        "device": 0.15,
        "preferred_time": 0.10
    }
    total_weight = sum(weights.values())
    combined = 0.0
    for name, val in score_components:
        combined += weights.get(name, 0) * val
    context_score = combined / total_weight
    # clamp
    context_score = max(0.0, min(1.0, context_score))
    return context_score, score_components

# ---------- Scoring & Recommendation ----------
def combined_score(sim_score, context_score, alpha=0.7):
    """
    Combine semantic similarity and context compatibility.
    alpha: weight for semantic similarity (0..1). final = alpha*sim + (1-alpha)*context
    """
    return float(alpha * sim_score + (1.0 - alpha) * context_score)

# ---------- Main App ----------
st.set_page_config(page_title="Context-Aware Course Recommender", layout="wide")
st.title("Context-Aware Course Recommender — EduMatch (Prototype)")

# Load data and resources
with st.spinner("Loading data and embeddings..."):
    df = load_df(CSV)
    emb_norm = load_embeddings(NPY)
    tfidf, tfidf_mat = build_tfidf(df)

# Sidebar: user context inputs
st.sidebar.header("User Context")
skill_level = st.sidebar.selectbox("Current skill level", ["Beginner", "Intermediate", "Advanced"])
hours_per_week = st.sidebar.slider("Available study time (hours per week)", 0.5, 40.0, 4.0, step=0.5)
device = st.sidebar.selectbox("Device used", ["Desktop", "Mobile"])
preferred_time = st.sidebar.selectbox("Preferred study time", ["Morning", "Afternoon", "Evening"])

st.sidebar.markdown("---")
st.sidebar.header("Algorithm settings")
alpha = st.sidebar.slider("Weight on semantic similarity (alpha)", 0.0, 1.0, 0.7, step=0.05)
use_transformer_for_query = st.sidebar.checkbox("Use Transformer for query encoding (better quality, slower)", value=False)
top_k = st.sidebar.slider("Top-K recommendations", 1, 15, DEFAULT_TOPK)
log_interactions = st.sidebar.checkbox("Log shown recommendations to local CSV", value=False)

# Compose user context dict
user_ctx = {
    "skill_level": skill_level,
    "hours_per_week": hours_per_week,
    "device": device,
    "preferred_time": preferred_time
}

# Mode selection
mode = st.radio("Recommendation Mode", ["By example course", "By text query"])

# Helper: get base similarity vector
def semantic_similarity_by_index(idx):
    return emb_norm.dot(emb_norm[idx])  # cosine because emb_norm is normalized

def semantic_similarity_by_text_tfidf(query):
    q = clean_text_raw(query)
    qv = tfidf.transform([q])
    sims = cosine_similarity(qv, tfidf_mat).ravel()
    # TF-IDF values are not normalized in same scale as embeddings. normalize to [0,1] for combination.
    # Simple min-max normalization across corpus (practical)
    vmin = sims.min()
    vmax = sims.max() if sims.max() != vmin else vmin + 1e-8
    sims_norm = (sims - vmin) / (vmax - vmin)
    return sims_norm

def semantic_similarity_by_text_transformer(query, transformer_model):
    q = clean_text_raw(query)
    q_emb = transformer_model.encode([q])
    q_emb_n = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    sims = emb_norm.dot(q_emb_n[0])
    # sims already cosine-like in [ -1,1 ], but practically [0,1]. Clip to [0,1]
    sims = np.clip(sims, 0.0, 1.0)
    return sims

# Recommendation flows
if mode == "By example course":
    st.subheader("Find courses similar to a selected course (context-aware)")
    course_choice = st.selectbox("Select a course", ["-- pick --"] + df["course_title"].astype(str).tolist())
    if course_choice != "-- pick --":
        idx = int(df.index[df["course_title"].astype(str) == course_choice][0])
        st.markdown("**Selected course:**")
        st.write(df.loc[idx, ["course_title","course_organization","course_rating","course_difficulty","course_Certificate_type"]])
        # compute semantic similarity vector
        sem_sim = semantic_similarity_by_index(idx)
        # compute context scores for all courses
        context_scores = []
        context_components = []
        for i, row in df.iterrows():
            cs, comps = context_score_for_course(user_ctx, row)
            context_scores.append(cs)
            context_components.append(comps)
        context_scores = np.array(context_scores)
        # compute combined scores
        final_scores = alpha * sem_sim + (1.0 - alpha) * context_scores
        # exclude self
        final_scores[idx] = -1.0
        top_idxs = np.argsort(-final_scores)[:top_k]
        st.subheader("Top recommendations (combined semantic + context)")
        for rank, i in enumerate(top_idxs, start=1):
            rscore = float(final_scores[i])
            cscore = float(context_scores[i])
            sscore = float(sem_sim[i])
            st.markdown(f"**{rank}. {df.loc[i,'course_title']}**")
            st.write(f"Org: {df.loc[i,'course_organization']}  | Rating: {df.loc[i,'course_rating']}  | Similarity: {sscore:.3f}  | Context score: {cscore:.3f}  | Final: {rscore:.3f}")
            st.write(f"Difficulty: {df.loc[i,'course_difficulty']}  | Certificate type: {df.loc[i,'course_Certificate_type']}")
            st.write("---")
        # optionally log
        if log_interactions:
            log_df = pd.DataFrame([{
                "mode":"by_example",
                "selected_index": int(idx),
                "selected_title": df.loc[idx,'course_title'],
                "user_skill": user_ctx['skill_level'],
                "hours_per_week": user_ctx['hours_per_week'],
                "device": user_ctx['device'],
                "preferred_time": user_ctx['preferred_time'],
                "alpha": alpha
            }])
            # append to local csv
            log_path = Path("interaction_log.csv")
            if log_path.exists():
                log_df.to_csv(log_path, mode='a', header=False, index=False)
            else:
                log_df.to_csv(log_path, index=False)
            st.info(f"Logged interaction to {log_path}")

else:
    st.subheader("Find courses from a text query (context-aware)")
    query = st.text_area("Enter a short description of what you want to learn", height=120)
    if st.button("Recommend"):
        if not query.strip():
            st.warning("Enter a query string.")
        else:
            # compute semantic similarity according to user's encoder choice
            if use_transformer_for_query:
                with st.spinner("Loading transformer model and encoding query (first time may be slow)..."):
                    transformer = load_transformer_model(TRANSFORMER_MODEL)
                sem_sim = semantic_similarity_by_text_transformer(query, transformer)
            else:
                sem_sim = semantic_similarity_by_text_tfidf(query)

            # compute context scores
            context_scores = np.array([context_score_for_course(user_ctx, row)[0] for _, row in df.iterrows()])

            # combine
            final_scores = alpha * sem_sim + (1.0 - alpha) * context_scores
            top_idxs = np.argsort(-final_scores)[:top_k]

            st.subheader("Top recommendations (combined semantic + context)")
            for rank, i in enumerate(top_idxs, start=1):
                rscore = float(final_scores[i])
                cscore = float(context_scores[i])
                sscore = float(sem_sim[i])
                st.markdown(f"**{rank}. {df.loc[i,'course_title']}**")
                st.write(f"Org: {df.loc[i,'course_organization']}  | Rating: {df.loc[i,'course_rating']}  | Similarity: {sscore:.3f}  | Context score: {cscore:.3f}  | Final: {rscore:.3f}")
                st.write(f"Difficulty: {df.loc[i,'course_difficulty']}  | Certificate type: {df.loc[i,'course_Certificate_type']}")
                st.write("---")

            # optionally log
            if log_interactions:
                log_df = pd.DataFrame([{
                    "mode":"by_query",
                    "query": query,
                    "user_skill": user_ctx['skill_level'],
                    "hours_per_week": user_ctx['hours_per_week'],
                    "device": user_ctx['device'],
                    "preferred_time": user_ctx['preferred_time'],
                    "alpha": alpha
                }])
                log_path = Path("interaction_log.csv")
                if log_path.exists():
                    log_df.to_csv(log_path, mode='a', header=False, index=False)
                else:
                    log_df.to_csv(log_path, index=False)
                st.info(f"Logged interaction to {log_path}")

# ---------- Footer: explanation & notes ----------
st.markdown("---")
st.subheader("How context influences recommendations (explainability)")
st.write("""
This prototype computes an interpretable **context score** per course from simple rules:
- Skill match (strongest weight): matches user's skill level to course difficulty.
- Time availability: approximated via course certificate type (SPECIALIZATION treated as long).
- Device: small boost if course appears mobile-friendly (title contains 'mobile'/'app').
- Preferred study time: not available in dataset (neutral).
Final ranking = alpha * semantic_similarity + (1-alpha) * context_score.
You can tune alpha in the sidebar to emphasize content vs context.
""")

st.markdown("**Notes & Limitations**")
st.write("""
- Context rules are coarse. For production, collect ground-truth logs and train a model (Factorization Machines, Contextual MF, or neural ranker) that ingests user, item, and context features.
- Better item metadata (explicit course duration, mobile compatibility, session timestamps) will significantly improve context matching.
- Privacy: when collecting device/time/location data, follow privacy policies and anonymize logs.
""")
