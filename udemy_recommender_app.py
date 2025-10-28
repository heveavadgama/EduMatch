# context_aware_app_fixed.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Lazy import placeholder for SentenceTransformer
SentenceTransformer = None

# ---------- Config ----------
CSV_COURSEA = "coursea_data.csv"
CSV_UDEMY = "udemy_courses_cleaned.csv"
EMB_FILE = "course_embeddings.npy"  # optional precomputed embeddings (for either dataset)
TFIDF_MAX_FEATURES = 4000

# ---------- Helpers ----------
def clean_text_raw(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+',' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return " ".join(text.split())

def detect_level_intent(query: str):
    q = str(query).lower()
    if any(w in q for w in ["advanced","expert","expert-level","pro"]):
        return "Advanced"
    if any(w in q for w in ["intermediate","intermediate-level"]):
        return "Intermediate"
    if any(w in q for w in ["beginner","intro","introduction","basic"]):
        return "Beginner"
    return None

def top_k_indices(arr, k=5, exclude_idx=None):
    a = np.array(arr, copy=True)
    if exclude_idx is not None:
        a[exclude_idx] = -np.inf
    idxs = np.argsort(-a)[:k]
    return idxs

# ---------- Load dataset (auto-detect udemy or coursera style) ----------
@st.cache_data(show_spinner=False)
def load_df():
    # prefer udemy cleaned if exists
    if Path(CSV_UDEMY).exists():
        df = pd.read_csv(CSV_UDEMY)
    elif Path(CSV_COURSEA).exists():
        df = pd.read_csv(CSV_COURSEA)
    else:
        st.error("No dataset found. Place coursea_data.csv or udemy_courses_cleaned.csv in repo root.")
        st.stop()
    # normalize some columns for common usage
    df.columns = [c.strip() for c in df.columns]
    # create clean_text if missing
    if 'clean_text' not in df.columns:
        df['text_data'] = (
            df.get('course_title','').fillna('') + ' ' +
            df.get('course_organization','').fillna('') + ' ' +
            df.get('course_Certificate_type','').fillna('') + ' ' +
            df.get('course_difficulty','').fillna('')
        )
        df['clean_text'] = df['text_data'].apply(clean_text_raw)
    # udemy specific: ensure content_duration_hr and level_simple exist
    if 'content_duration' in df.columns and 'content_duration_hr' not in df.columns:
        def parse_duration(x):
            try:
                return float(x)
            except:
                return np.nan
        df['content_duration_hr'] = df['content_duration'].apply(parse_duration)
    if 'level_simple' not in df.columns:
        # try to map common level column names
        if 'course_difficulty' in df.columns:
            df['level_simple'] = df['course_difficulty'].astype(str).replace('nan','')
        elif 'level' in df.columns:
            def map_level(l):
                l = str(l).lower()
                if 'beginner' in l: return 'Beginner'
                if 'intermediate' in l: return 'Intermediate'
                if 'advanced' in l: return 'Advanced'
                if 'all' in l: return 'All'
                return 'Unknown'
            df['level_simple'] = df['level'].apply(map_level)
        else:
            df['level_simple'] = 'Unknown'
    # fill missing display columns
    for c in ['course_title','course_organization','course_rating']:
        if c not in df.columns:
            df[c] = ''
    return df.reset_index(drop=True)

# ---------- Embeddings & TF-IDF ----------
@st.cache_resource(show_spinner=False)
def load_embeddings(path=EMB_FILE):
    p = Path(path)
    if not p.exists():
        return None
    emb = np.load(p)
    # normalize
    emb_norm = emb.astype('float32')
    norms = np.linalg.norm(emb_norm, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    emb_norm = emb_norm / norms
    return emb_norm

@st.cache_resource(show_spinner=False)
def build_tfidf(df):
    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words='english')
    mat = tfidf.fit_transform(df['clean_text'].astype(str))
    return tfidf, mat

@st.cache_resource(show_spinner=False)
def load_transformer(name="paraphrase-MiniLM-L3-v2"):
    global SentenceTransformer
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as ST
        SentenceTransformer = ST
    return SentenceTransformer(name)

# ---------- Scoring components ----------
def improved_skill_score(user_skill, course_level):
    # stricter scoring: exact match=1, adjacent=0.5, opposite=0.0
    mapping = {'Beginner':0,'All':0,'Intermediate':1,'Advanced':2,'Unknown':1}
    us = {'Beginner':0,'Intermediate':1,'Advanced':2}.get(user_skill,1)
    cs = mapping.get(course_level,1)
    dist = abs(us - cs)
    if dist == 0:
        return 1.0
    if dist == 1:
        return 0.5
    return 0.0

def time_score_continuous(user_hours_per_week, course_total_hours, expected_weeks=4):
    try:
        ch = float(course_total_hours)
        if np.isnan(ch) or ch <= 0:
            return 0.5  # neutral
        required_weekly = max(0.5, ch / expected_weeks)
        score = user_hours_per_week / required_weekly
        return float(max(0.0, min(1.0, score)))
    except:
        return 0.5

# ---------- Main app ----------
st.set_page_config(page_title="Context-Aware Course Recommender — Fixed", layout="wide")
st.title("Context-Aware Course Recommender — Fixed & Hardened")

df = load_df()
emb_norm = load_embeddings()
tfidf, tfidf_mat = build_tfidf(df)

# Sidebar controls
st.sidebar.header("User Context")
skill_level = st.sidebar.selectbox("Your skill level", ["Beginner","Intermediate","Advanced"], index=1)
hours_per_week = st.sidebar.slider("Available study time (hours/week)", 0.5, 40.0, 4.0, step=0.5)
device = st.sidebar.selectbox("Device used", ["Desktop","Mobile"])
preferred_time = st.sidebar.selectbox("Preferred study time", ["Morning","Afternoon","Evening"])

st.sidebar.markdown("---")
st.sidebar.header("Recommendation Settings")
alpha = st.sidebar.slider("Weight on semantic similarity (alpha)", 0.0, 1.0, 0.75, step=0.05)
use_transformer = st.sidebar.checkbox("Use Transformer for query encoding (better quality)", value=True)
desired_level_ui = st.sidebar.selectbox("Preferred difficulty (UI override)", ["Any", "Beginner", "Intermediate", "Advanced"], index=0)
strict_difficulty = st.sidebar.checkbox("Require difficulty match (strict filter)", value=False)
soft_boost = st.sidebar.checkbox("Apply soft boost for preferred difficulty", value=True)
boost_factor = st.sidebar.slider("Difficulty boost factor (if soft)", 1.0, 2.0, 1.25, step=0.05)
top_k = st.sidebar.slider("Top-K", 1, 20, 5)

st.sidebar.markdown("---")
st.sidebar.write("Intent detection from query is enabled. If query contains 'advanced' it will be used as preferred difficulty (unless UI override).")

# UI: mode and query
mode = st.radio("Mode", ["By text query", "By example course"])
query = ""
if mode == "By text query":
    query = st.text_area("Enter your query (e.g., 'advanced python web development')", height=120)

# Determine desired difficulty
detected_level = detect_level_intent(query) if query else None
desired_level = desired_level_ui if desired_level_ui != "Any" else (detected_level or None)

# Helper: semantic similarity functions
def semantic_sim_by_query(query_text):
    q = clean_text_raw(query_text)
    if use_transformer and emb_norm is not None:
        # lazy-load model
        try:
            model = load_transformer()
            q_emb = model.encode([q])
            q_emb_n = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
            sims = emb_norm.dot(q_emb_n[0])
            sims = np.clip(sims, 0.0, 1.0)
            return sims
        except Exception as e:
            # fallback to TF-IDF
            st.warning(f"Transformer encoding failed, falling back to TF-IDF. ({e})")
    # TF-IDF fallback
    qv = tfidf.transform([q])
    sims = cosine_similarity(qv, tfidf_mat).ravel()
    mn, mx = sims.min(), sims.max()
    if mx - mn < 1e-9:
        return np.zeros_like(sims)
    return (sims - mn) / (mx - mn)

def semantic_sim_by_index(idx):
    if emb_norm is not None:
        sims = emb_norm.dot(emb_norm[idx])
        return np.clip(sims, 0.0, 1.0)
    else:
        v = tfidf_mat[idx]
        sims = cosine_similarity(v, tfidf_mat).ravel()
        mn, mx = sims.min(), sims.max()
        if mx - mn < 1e-9:
            return np.zeros_like(sims)
        return (sims - mn) / (mx - mn)

# Prepare per-course context base scores
def compute_context_scores(user_ctx):
    # returns numpy array of context scores in [0,1]
    skill_arr = np.array([improved_skill_score(user_ctx['skill_level'], lvl) for lvl in df['level_simple']])
    # time scores: use content_duration_hr if present else neutral
    if 'content_duration_hr' in df.columns:
        time_arr = np.array([time_score_continuous(user_ctx['hours_per_week'], h) for h in df['content_duration_hr'].fillna(np.nan)])
    else:
        time_arr = np.array([0.5]*len(df))
    # device score: basic heuristic
    title_lower = df['course_title'].astype(str).str.lower()
    mobile_friendly = np.where(title_lower.str.contains('mobile|app|responsive'), 1.0, 0.95)
    device_arr = np.where(user_ctx['device'].lower()=="mobile", mobile_friendly, 1.0)
    # combine with weights
    w_skill, w_time, w_device = 0.5, 0.3, 0.2
    combined = w_skill*skill_arr + w_time*time_arr + w_device*device_arr
    # clip
    combined = np.clip(combined, 0.0, 1.0)
    return combined, {'skill':skill_arr, 'time':time_arr, 'device':device_arr}

# Candidate selection based on desired difficulty
def candidate_indices_for_difficulty(desired_level, strict):
    if not desired_level:
        return np.arange(len(df))
    mask = df['level_simple'] == desired_level
    idxs = np.where(mask)[0]
    if strict:
        if len(idxs) == 0:
            # no candidates of that level, fallback to all (user can disable strict)
            return np.arange(len(df))
        return idxs
    else:
        return np.arange(len(df))  # soft handled via boosting

# Apply soft boost multiplier
def apply_difficulty_boost(final_scores, desired_level, boost_factor):
    if not desired_level or not boost_factor or boost_factor <= 1.0:
        return final_scores
    mult = np.where(df['level_simple'] == desired_level, boost_factor, 1.0)
    return final_scores * mult

# Main recommendation flow
user_ctx = {'skill_level': skill_level, 'hours_per_week': hours_per_week, 'device': device, 'preferred_time': preferred_time}
context_scores_all, context_components = compute_context_scores(user_ctx)

if mode == "By text query":
    if st.button("Recommend"):
        if not query.strip():
            st.warning("Enter a query.")
        else:
            sem_sim_all = semantic_sim_by_query(query)
            # candidate indices based on strict desired_level
            candidate_idxs = candidate_indices_for_difficulty(desired_level, strict_difficulty)
            # compute final for candidates
            sem_sim = sem_sim_all[candidate_idxs]
            context_for_candidates = context_scores_all[candidate_idxs]
            final = alpha * sem_sim + (1.0 - alpha) * context_for_candidates
            # apply soft boosting if enabled and desired_level set
            if soft_boost and desired_level:
                # apply boost factor aligned to full df then select subset
                mult = np.where(df['level_simple'] == desired_level, boost_factor, 1.0)
                final = final * mult[candidate_idxs]
            # select topK within candidate set
            top_local = np.argsort(-final)[:top_k]
            top_global = candidate_idxs[top_local]
            st.subheader(f"Top {top_k} results (desired: {desired_level or 'Any'})")
            for rank, i in enumerate(top_global, start=1):
                sscore = float(sem_sim_all[i])
                cscore = float(context_scores_all[i])
                fscore = float(alpha*sscore + (1-alpha)*cscore)
                st.markdown(f"**{rank}. {df.loc[i,'course_title']}**")
                st.write(f"Subject/Org: {df.loc[i].get('subject', df.loc[i].get('course_organization',''))}  |  Duration: {df.loc[i].get('content_duration_hr', '')}  | Level: {df.loc[i,'level_simple']}")
                st.write(f"Semantic: {sscore:.3f}  | Context: {cscore:.3f}  | Combined: {fscore:.3f}")
                st.write("---")

else:
    # By example course
    st.subheader("Pick an example course")
    choice = st.selectbox("Course", ["-- pick --"] + df['course_title'].astype(str).tolist())
    if choice != "-- pick --":
        idx = int(df.index[df['course_title'].astype(str) == choice][0])
        # base semantic similarity
        sem_all = semantic_sim_by_index(idx)
        candidate_idxs = candidate_indices_for_difficulty(desired_level, strict_difficulty)
        sem_sim = sem_all[candidate_idxs]
        context_for_candidates = context_scores_all[candidate_idxs]
        final = alpha * sem_sim + (1.0 - alpha) * context_for_candidates
        if soft_boost and desired_level:
            mult = np.where(df['level_simple'] == desired_level, boost_factor, 1.0)
            final = final * mult[candidate_idxs]
        # exclude self if present
        # map idx to candidate index position
        if idx in candidate_idxs:
            pos = np.where(candidate_idxs==idx)[0]
            if len(pos)>0:
                final[pos[0]] = -np.inf
        top_local = np.argsort(-final)[:top_k]
        top_global = candidate_idxs[top_local]
        st.subheader(f"Top {top_k} similar courses (desired: {desired_level or 'Any'})")
        for rank, i in enumerate(top_global, start=1):
            sscore = float(sem_all[i])
            cscore = float(context_scores_all[i])
            fscore = float(alpha*sscore + (1-alpha)*cscore)
            st.markdown(f"**{rank}. {df.loc[i,'course_title']}**")
            st.write(f"Subject/Org: {df.loc[i].get('subject', df.loc[i].get('course_organization',''))}  |  Duration: {df.loc[i].get('content_duration_hr','')}  | Level: {df.loc[i,'level_simple']}")
            st.write(f"Semantic: {sscore:.3f}  | Context: {cscore:.3f}  | Combined: {fscore:.3f}")
            st.write("---")

# Explainability panel
st.markdown("---")
st.subheader("Explainability & How matches were computed")
st.write("""
1. Semantic similarity computed by Transformer embeddings (if available) or TF-IDF as fallback.
2. Context score computed from:
   - Skill match (strict scoring, exact=1, adjacent=0.5, opposite=0)
   - Time availability (continuous match using course duration if present)
   - Device compatibility heuristic
3. Final score = alpha * semantic + (1-alpha) * context.
4. Difficulty preference: can be enforced strictly or applied as a soft boost multiplier.
""")
