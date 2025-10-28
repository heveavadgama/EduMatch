"""
Streamlit: Context-Aware Udemy Recommender

Context: skill_level, time_availability_hours_per_week, device, preferred_study_time.
Model: hybrid re-ranker = content similarity (embeddings or TF-IDF) + popularity/quality
       + duration fit + device fit + LinUCB contextual bandit.
Guards: strict difficulty gating to stop beginner drift for advanced users.
"""

import os
import re
import math
import json
import pickle
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="Context-Aware Udemy Recommender", layout="wide")
st.title("Context-Aware Course Recommender")

DATA_CSV = "udemy_courses_cleaned.csv"  # fallback to udemy_courses.csv
EMB_MODULE = "udemy_course_embeddings"   # optional local embedding helper: exposes get_course_embeddings(df), get_query_embedding(text)
STATE_DIR = ".reco_state"
MODEL_FILE = os.path.join(STATE_DIR, "linucb.pkl")
os.makedirs(STATE_DIR, exist_ok=True)

# ---------------------------
# Data loading
# ---------------------------
@st.cache_data(show_spinner=False)
def load_courses() -> pd.DataFrame:
    path = DATA_CSV if os.path.exists(DATA_CSV) else "udemy_courses.csv"
    df = pd.read_csv(path)

    # Harmonize common schemas
    rename_map = {}
    if "course_title" in df.columns and "title" not in df.columns:
        rename_map["course_title"] = "title"
    if "course_rating" in df.columns and "avg_rating" not in df.columns:
        rename_map["course_rating"] = "avg_rating"
    if "rating" in df.columns and "avg_rating" not in df.columns:
        rename_map["rating"] = "avg_rating"
    if "link" in df.columns and "url" not in df.columns:
        rename_map["link"] = "url"
    if "duration" in df.columns and "content_duration" not in df.columns:
        rename_map["duration"] = "content_duration"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Expected columns. Create if missing.
    expected = [
        "course_id", "title", "url", "price", "num_subscribers",
        "avg_rating", "num_reviews", "num_lectures", "level",
        "content_duration", "published_timestamp", "subject", "description"
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan

    # Clean
    df["level"] = df["level"].fillna("Unknown")
    df["content_duration"] = pd.to_numeric(df["content_duration"], errors="coerce").fillna(0.0)
    df["avg_rating"] = pd.to_numeric(df["avg_rating"], errors="coerce")
    df["num_reviews"] = pd.to_numeric(df["num_reviews"], errors="coerce").fillna(0.0)
    df["num_subscribers"] = pd.to_numeric(df["num_subscribers"], errors="coerce").fillna(0.0)
    df["num_lectures"] = pd.to_numeric(df["num_lectures"], errors="coerce").fillna(0.0)
    df["description"] = df.get("description", "").fillna("")
    df["subject"] = df.get("subject", "").fillna("")
    if "published_timestamp" in df.columns:
        df["published_timestamp"] = pd.to_datetime(df["published_timestamp"], errors="coerce")
    else:
        df["published_timestamp"] = pd.NaT

    # Difficulty map
    level_map = {
        "Beginner": 0, "Beginner Level": 0,
        "All Levels": 0.5, "All": 0.5,
        "Intermediate": 1, "Intermediate Level": 1,
        "Expert": 2, "Expert Level": 2, "Advanced": 2,
        "Unknown": 0.5,
    }
    df["level_num"] = df["level"].map(level_map).fillna(0.5)

    # Canonical text
    base_text_cols = ["title", "subject", "description"]
    extra_text_candidates = ["headline", "objectives", "what_you_will_learn", "requirements"]
    extra_present = [c for c in extra_text_candidates if c in df.columns]
    text_cols = [c for c in base_text_cols + extra_present if c in df.columns]
    if text_cols:
        df[text_cols] = df[text_cols].astype(str).replace("nan", "", regex=False)
        df["text"] = df[text_cols].agg(" \n".join, axis=1)
    else:
        df["text"] = ""

    # Fill missing titles to avoid 'nan'
    missing_title = df["title"].isna() | (df["title"].astype(str).str.strip() == "")
    if "course_id" in df.columns:
        fallback_ids = df["course_id"].fillna(0).astype(int).astype(str)
    else:
        fallback_ids = pd.Series(range(len(df))).astype(str)
    df.loc[missing_title, "title"] = "Course #" + fallback_ids[missing_title]

    return df

courses = load_courses()

# ---------------------------
# Embeddings
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    """Import user's embedding helper or fallback to TF-IDF."""
    try:
        imp = __import__(EMB_MODULE)
        # Expect: get_course_embeddings(df)->np.ndarray, get_query_embedding(text)->np.ndarray
        # They can return sparse; we will densify.
        return imp
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        vec.fit(courses["text"].tolist())

        class _Fallback:
            def get_course_embeddings(self, df):
                return vec.transform(df["text"].tolist()).astype(np.float32)
            def get_query_embedding(self, text):
                return vec.transform([text]).astype(np.float32)

        return _Fallback()

embedder = load_embedder()

@st.cache_data(show_spinner=False)
def get_course_embeddings() -> np.ndarray:
    embs = embedder.get_course_embeddings(courses)
    if not isinstance(embs, np.ndarray):
        try:
            embs = embs.toarray()
        except Exception:
            embs = np.asarray(embs)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    return embs / norms

COURSE_EMB = get_course_embeddings()

# ---------------------------
# LinUCB contextual bandit
# ---------------------------
class LinUCB:
    def __init__(self, d: int, alpha: float = 0.2):
        self.d = d
        self.alpha = alpha
        self.A = np.eye(d)
        self.b = np.zeros((d, 1))
    def score(self, x: np.ndarray) -> float:
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        x = x.reshape(-1, 1)
        mean = float((x.T @ theta).ravel())
        bonus = self.alpha * math.sqrt(float(x.T @ A_inv @ x))
        return mean + bonus
    def update(self, x: np.ndarray, reward: float):
        x = x.reshape(-1, 1)
        self.A += x @ x.T
        self.b += reward * x

def load_or_init_bandit(d: int) -> LinUCB:
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, LinUCB) and obj.d == d:
                return obj
        except Exception:
            pass
    return LinUCB(d=d, alpha=0.2)

def save_bandit(b: LinUCB):
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(b, f)

# ---------------------------
# Context encoders
# ---------------------------
def encode_context(skill: str, hours_per_week: int, device: str, study_time: str) -> np.ndarray:
    skill_map = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
    device_map = {"Mobile": [1, 0], "Desktop": [0, 1]}
    time_map = {"Morning": [1, 0], "Evening": [0, 1]}
    x = [skill_map.get(skill, 1), hours_per_week]
    x += device_map.get(device, [0, 1])
    x += time_map.get(study_time, [1, 0])
    return np.array(x, dtype=float)

# ---------------------------
# Scoring components
# ---------------------------
@st.cache_data(show_spinner=False)
def popularity_scores(df: pd.DataFrame) -> np.ndarray:
    cols = ["avg_rating", "num_reviews", "num_subscribers"]
    X = df[cols].copy()

    # If rating fully missing, drop from score
    if X["avg_rating"].isna().all():
        X["avg_rating"] = 0.0
        rating_weight = 0.0
    else:
        X["avg_rating"] = X["avg_rating"].fillna(X["avg_rating"].median())
        rating_weight = 0.4

    X["num_reviews"] = pd.to_numeric(X["num_reviews"], errors="coerce").fillna(0.0)
    X["num_subscribers"] = pd.to_numeric(X["num_subscribers"], errors="coerce").fillna(0.0)

    mms = MinMaxScaler()
    Xn = mms.fit_transform(X)

    if rating_weight == 0:
        pop = 0.6 * Xn[:, 1] + 0.4 * Xn[:, 2]
    else:
        pop = rating_weight * Xn[:, 0] + 0.35 * Xn[:, 1] + 0.25 * Xn[:, 2]
    return pop

POP = popularity_scores(courses)

def duration_fit(df: pd.DataFrame, hours_per_week: int, horizon_weeks: int = 6) -> np.ndarray:
    cap = max(1, hours_per_week * horizon_weeks)
    dur = df["content_duration"].fillna(0.0).to_numpy()
    fit = np.where(dur <= cap, 1.0, np.clip(cap / (dur + 1e-6), 0.1, 1.0))
    return fit

def device_fit(df: pd.DataFrame, device: str) -> np.ndarray:
    if device == "Mobile":
        mms = MinMaxScaler()
        two = mms.fit_transform(df[["num_lectures", "content_duration"]].fillna(0.0))
        return 1.0 - 0.5 * two[:, 0] - 0.5 * two[:, 1]
    return np.ones(len(df))

def difficulty_guard(df: pd.DataFrame, user_skill: str, strict: bool = True) -> np.ndarray:
    if not strict:
        return np.ones(len(df))
    us = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}.get(user_skill, 1)
    lv = df["level_num"].to_numpy()
    lvl_str = df["level"].astype(str).str.lower().fillna("")
    allow_all = lvl_str.str.contains("all").to_numpy()
    if us >= 2:  # Advanced
        mask = ((lv >= 1) | allow_all).astype(float)
    elif us == 1:  # Intermediate
        mask = ((lv >= 0.5) | allow_all).astype(float)
    else:
        mask = np.ones(len(df))
    return mask

def hybrid_score(
    df: pd.DataFrame,
    query_vec: np.ndarray,
    hours_per_week: int,
    device: str,
    user_skill: str,
    ctx_vec: np.ndarray,
    bandit: LinUCB,
    strict_gate: bool,
    weights=None,
) -> np.ndarray:
    # Dynamic weights: upweight content when ratings are scarce
    if weights is None:
        w = 0.6 if pd.isna(df["avg_rating"]).all() else 0.45
        weights = (w, 0.25, 0.1, 0.05)
    a, b, c, d = weights

    # Content similarity
    sim = cosine_similarity(query_vec, COURSE_EMB).ravel()
    if sim.max() - sim.min() > 1e-9:
        sim = (sim - sim.min()) / (sim.max() - sim.min())

    pop = POP
    dur = duration_fit(df, hours_per_week)
    dev = device_fit(df, device)
    base = a * sim + b * pop + c * dur + d * dev

    # Difficulty guard
    guard = difficulty_guard(df, user_skill, strict=strict_gate)
    base = base * guard

    # Bandit adjustment using simple item meta
    meta = np.stack([
        df["level_num"].to_numpy(),
        np.log1p(df["content_duration"].fillna(0.0).to_numpy()),
        np.log1p(df["num_lectures"].fillna(0.0).to_numpy()),
        np.ones(len(df)),
    ], axis=1)
    bandit_scores = np.array([bandit.score(np.concatenate([ctx_vec, m])) for m in meta])
    if bandit_scores.max() - bandit_scores.min() > 1e-9:
        band_adj = 0.2 * (bandit_scores - bandit_scores.min()) / (bandit_scores.max() - bandit_scores.min())
    else:
        band_adj = np.zeros_like(bandit_scores)

    return base + band_adj

# ---------------------------
# UI
# ---------------------------
with st.sidebar:
    st.header("Your Context")
    skill = st.selectbox("Current skill level", ["Beginner", "Intermediate", "Advanced"], index=1)
    hours = st.slider("Time availability (hours/week)", 1, 40, 6)
    device = st.selectbox("Device used", ["Mobile", "Desktop"], index=1)
    pref_time = st.selectbox("Preferred study time", ["Morning", "Evening"], index=0)
    query = st.text_input("Interests/keywords", value="python data analysis pandas")
    require_match = st.checkbox("Require keyword match", value=True, help="Only show items containing at least one query token.")
    k = st.slider("How many recommendations", 5, 30, 10)
    strict_gate = st.checkbox("Strict difficulty gating", value=True, help="Blocks Beginner items for Intermediate+ and blocks below-Intermediate for Advanced. 'All Levels' always allowed.")

# Context vector and bandit
ctx = encode_context(skill, hours, device, pref_time)
bandit = load_or_init_bandit(d=len(ctx) + 4)  # + item meta dims

# Query embedding
qv = embedder.get_query_embedding(query)
if not isinstance(qv, np.ndarray):
    try:
        qv = qv.toarray()
    except Exception:
        qv = np.asarray(qv)
qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-8)

# Score
scores = hybrid_score(courses, qv, hours, device, skill, ctx, bandit, strict_gate)
rank_idx = np.argsort(-scores)

# Optional keyword filter
if require_match:
    tokens = [t.strip().lower() for t in query.split() if t.strip()]
    if tokens:
        text_series = courses["text"].astype(str).str.lower()
        pattern = "|".join(re.escape(t) for t in tokens)
        kw_mask = text_series.str.contains(pattern, regex=True, na=False)
        candidate_idx = [i for i in rank_idx if kw_mask.iloc[i]]
    else:
        candidate_idx = rank_idx.tolist()
else:
    candidate_idx = rank_idx.tolist()

rec_idx = [i for i in candidate_idx if np.isfinite(scores[i])][: 3 * k]

# Final post-filters: keep valid titles
mask_ok = courses.iloc[rec_idx]["title"].astype(str).str.strip().ne("")
rec_idx = list(np.array(rec_idx)[np.where(mask_ok)[0]])

# Diversity by subject
seen_subjects = set()
final = []
for i in rec_idx:
    subj = str(courses.iloc[i]["subject"]).strip().lower()
    if subj not in seen_subjects or len(final) < max(1, k // 2):
        final.append(i)
        seen_subjects.add(subj)
    if len(final) >= k:
        break
if len(final) < k:
    final += rec_idx[: k - len(final)]
final = final[:k]

st.subheader("Recommendations")

def safe_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default

def render_row(row: pd.Series, score: float, idx: int):
    cols = st.columns([6, 2, 2, 2])
    with cols[0]:
        ttl = str(row["title"]).strip() or f"Course #{safe_int(row.get('course_id', idx), idx)}"
        st.markdown(f"**{ttl}**")
        url = str(row.get("url", "")).strip()
        if url:
            st.markdown(f"[Open course]({url})")
        st.caption(f"Level: {row['level']} • Subject: {row['subject']} • Duration: {row['content_duration']}h • Lectures: {safe_int(row['num_lectures'])}")
    with cols[1]:
        rating_val = row.get("avg_rating")
        if pd.isna(rating_val) or float(rating_val) == 0.0:
            st.metric("Rating", "—")
        else:
            st.metric("Rating", f"{float(rating_val):.2f}")
    with cols[2]:
        st.metric("Reviews", safe_int(row.get("num_reviews", 0)))
    with cols[3]:
        st.markdown(f"Score: {score:.3f}")

    # Feedback
    fb_cols = st.columns(3)
    meta_vec = np.array([
        row["level_num"],
        math.log1p(float(row.get("content_duration", 0.0))),
        math.log1p(float(row.get("num_lectures", 0.0))),
        1.0,
    ])
    with fb_cols[0]:
        if st.button("👍 Helpful", key=f"up_{idx}"):
            x = np.concatenate([ctx, meta_vec])
            bandit.update(x, reward=1.0)
            save_bandit(bandit)
            st.experimental_rerun()
    with fb_cols[1]:
        if st.button("👎 Not for me", key=f"down_{idx}"):
            x = np.concatenate([ctx, meta_vec])
            bandit.update(x, reward=-0.5)
            save_bandit(bandit)
            st.experimental_rerun()
    with fb_cols[2]:
        if st.button("🚫 Too easy", key=f"ban_{idx}"):
            x = np.concatenate([ctx, meta_vec])
            bandit.update(x, reward=-1.0)
            save_bandit(bandit)
            st.experimental_rerun()

# Render
if len(final) == 0:
    # Backoff: relax gating and keyword filter
    st.warning("No items matched after constraints. Relaxing gating and keyword filter.")
    scores_relaxed = hybrid_score(courses, qv, hours, device, skill, ctx, bandit, strict_gate=False)
    rank_idx = np.argsort(-scores_relaxed)
    rec_idx = [i for i in rank_idx if np.isfinite(scores_relaxed[i])][: 3 * k]
    if len(rec_idx) == 0:
        st.info("Still no items. Check CSV columns like 'title', 'level', 'content_duration'.")
    else:
        final = rec_idx[:k]
        for _, i in enumerate(final, 1):
            r = courses.iloc[i]
            render_row(r, float(scores_relaxed[i]), i)
else:
    for _, i in enumerate(final, 1):
        r = courses.iloc[i]
        render_row(r, float(scores[i]), i)

# ---------------------------
# Diagnostics
# ---------------------------
with st.expander("Diagnostics"):
    st.write("Context vector:", ctx.tolist())
    st.write("Bandit dims:", bandit.d)
    st.write({
        "rows": len(courses),
        "non_null_titles": int(courses["title"].astype(str).str.strip().ne("").sum()),
        "levels_sample": courses["level"].dropna().astype(str).str.lower().value_counts().head(10).to_dict(),
        "ratings_missing_all": bool(courses["avg_rating"].isna().all())
    })
    st.write("Guard: Beginner items blocked for Intermediate+, unless course is 'All Levels'.")
    st.write("Tip: use focused keywords to steer content similarity.")
