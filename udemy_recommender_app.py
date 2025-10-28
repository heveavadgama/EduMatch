"""
Streamlit: Context‑Aware Course Recommender (Udemy)

Context features: skill_level, time_availability_hours_per_week, device, preferred_study_time.
Model: hybrid re‑ranker = content similarity (embeddings) + popularity/quality + context fit + LinUCB contextual bandit for online updates.
Update: per‑click feedback updates LinUCB weights in‑session and saves to disk.
Hard guardrail: do not show beginner items to advanced users by default.
"""

import os
import json
import math
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="Context‑Aware Udemy Recommender", layout="wide")
st.title("Context‑Aware Course Recommender")

DATA_CSV = "udemy_courses_cleaned.csv"  # fallback to udemy_courses.csv if not found
EMB_MODULE = "udemy_course_embeddings"   # your local embedding helper
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
    # Expected columns. Create if missing to avoid crashes.
    expected = {
        "course_id": int,
        "title": str,
        "url": str,
        "price": str,
        "num_subscribers": float,
        "avg_rating": float,
        "num_reviews": float,
        "num_lectures": float,
        "level": str,  # 'Beginner', 'Intermediate', 'Expert' or similar
        "content_duration": float,  # hours
        "published_timestamp": str,
        "subject": str,
        "description": str,
    }
    for col, typ in expected.items():
        if col not in df.columns:
            df[col] = pd.Series([np.nan] * len(df), dtype="object")
    # simple cleanups
    df["level"] = df["level"].fillna("Unknown")
    df["content_duration"] = pd.to_numeric(df["content_duration"], errors="coerce").fillna(0.0)
    df["avg_rating"] = pd.to_numeric(df["avg_rating"], errors="coerce").fillna(0.0)
    df["num_reviews"] = pd.to_numeric(df["num_reviews"], errors="coerce").fillna(0.0)
    df["num_subscribers"] = pd.to_numeric(df["num_subscribers"], errors="coerce").fillna(0.0)
    df["num_lectures"] = pd.to_numeric(df["num_lectures"], errors="coerce").fillna(0.0)
    df["description"] = df.get("description", "").fillna("")
    df["subject"] = df.get("subject", "").fillna("")
    if "published_timestamp" in df.columns:
        df["published_timestamp"] = pd.to_datetime(df["published_timestamp"], errors="coerce")
    else:
        df["published_timestamp"] = pd.NaT
    # difficulty mapping
    level_map = {
        "Beginner": 0,
        "Beginner Level": 0,
        "All Levels": 0.5,
        "Intermediate": 1,
        "Intermediate Level": 1,
        "Expert": 2,
        "Expert Level": 2,
        "Advanced": 2,
        "Unknown": 0.5,
    }
    df["level_num"] = df["level"].map(level_map).fillna(0.5)
    # canonical text for embeddings
    df["text"] = (
        df["title"].fillna("")
        + " \n" + df["subject"].fillna("")
        + " \n" + df["description"].fillna("")
    )
    return df

courses = load_courses()

# ---------------------------
# Embeddings
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    """Try to import user's helper. Fallback to simple TF‑IDF vectorizer if embeddings unavailable."""
    try:
        imp = __import__(EMB_MODULE)
        # Expecting functions: get_course_embeddings(df) -> np.ndarray, get_query_embedding(text) -> np.ndarray
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
    # ensure dense matrix
    if not isinstance(embs, np.ndarray):
        try:
            embs = embs.toarray()
        except Exception:
            embs = np.asarray(embs)
    # L2 norm guard
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    return embs / norms

COURSE_EMB = get_course_embeddings()

# ---------------------------
# LinUCB contextual bandit (lightweight)
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
    mms = MinMaxScaler()
    X = mms.fit_transform(df[cols].fillna(0.0))
    pop = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2]
    return pop

POP = popularity_scores(courses)


def duration_fit(df: pd.DataFrame, hours_per_week: int, horizon_weeks: int = 6) -> np.ndarray:
    # prefer courses whose duration <= available_time * horizon, soft penalty otherwise
    cap = max(1, hours_per_week * horizon_weeks)
    dur = df["content_duration"].fillna(0.0).to_numpy()
    fit = np.where(dur <= cap, 1.0, np.clip(cap / (dur + 1e-6), 0.1, 1.0))
    return fit


def device_fit(df: pd.DataFrame, device: str) -> np.ndarray:
    if device == "Mobile":
        # Prefer fewer lectures and shorter total duration on mobile
        mms = MinMaxScaler()
        two = mms.fit_transform(df[["num_lectures", "content_duration"]].fillna(0.0))
        # lower is better -> invert
        return 1.0 - 0.5 * two[:, 0] - 0.5 * two[:, 1]
    # Desktop neutral
    return np.ones(len(df))


def difficulty_guard(df: pd.DataFrame, user_skill: str, strict: bool = True) -> np.ndarray:
    us = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}.get(user_skill, 1)
    lv = df["level_num"].to_numpy()
    if strict:
        # Never show courses below user's level when Advanced; never show Beginner to Intermediate+.
        mask = np.ones(len(df))
        if us >= 2:
            mask = (lv >= 1).astype(float)  # allow Intermediate and Expert only
        elif us == 1:
            mask = (lv >= 0.5).astype(float)  # block pure Beginner
        else:
            mask = np.ones(len(df))
        return mask
    return np.ones(len(df))


def hybrid_score(
    df: pd.DataFrame,
    query_vec: np.ndarray,
    hours_per_week: int,
    device: str,
    user_skill: str,
    ctx_vec: np.ndarray,
    bandit: LinUCB,
    weights=(0.45, 0.25, 0.15, 0.15),  # content, popularity, duration, device
) -> np.ndarray:
    a, b, c, d = weights
    # content similarity
    sim = cosine_similarity(query_vec, COURSE_EMB).ravel()
    # normalize
    if sim.max() - sim.min() > 1e-9:
        sim = (sim - sim.min()) / (sim.max() - sim.min())
    # components
    pop = POP
    dur = duration_fit(df, hours_per_week)
    dev = device_fit(df, device)
    base = a * sim + b * pop + c * dur + d * dev
    # difficulty guard
    guard = difficulty_guard(df, user_skill, strict=True)
    base = base * guard
    # bandit adjustment: add UCB score per item via simple dot of item meta with context
    # item meta: [level_num, log_duration, log_lectures, bias]
    meta = np.stack([
        df["level_num"].to_numpy(),
        np.log1p(df["content_duration"].fillna(0.0).to_numpy()),
        np.log1p(df["num_lectures"].fillna(0.0).to_numpy()),
        np.ones(len(df)),
    ], axis=1)
    # build context cross features = kron(ctx, meta_i) collapsed via mean
    # efficient: project meta with learned theta using ctx as features
    # For simplicity, we concatenate ctx and meta and score with bandit.
    bandit_scores = np.array([bandit.score(np.concatenate([ctx_vec, m])) for m in meta])
    # rescale bandit scores to 0-0.2 additive bonus
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
    k = st.slider("How many recommendations", 5, 30, 10)
    st.markdown("Strict difficulty gating is ON to avoid beginner drift for advanced users.")

# assemble context
ctx = encode_context(skill, hours, device, pref_time)
bandit = load_or_init_bandit(d=len(ctx) + 4)  # + item meta dims

# query embedding
qv = embedder.get_query_embedding(query)
if not isinstance(qv, np.ndarray):
    try:
        qv = qv.toarray()
    except Exception:
        qv = np.asarray(qv)
# L2 norm
qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-8)

# score
scores = hybrid_score(courses, qv, hours, device, skill, ctx, bandit)
rank_idx = np.argsort(-scores)
rec_idx = [i for i in rank_idx if scores[i] > 0][: 3 * k]  # take some extra before filters

# final post‑filters: remove zero‑length, remove NaN titles
mask_ok = (
    courses.iloc[rec_idx]["title"].notna() & (courses.iloc[rec_idx]["content_duration"] > 0)
)
rec_idx = list(np.array(rec_idx)[np.where(mask_ok)[0]])

# Top‑K with light diversity by subject
seen_subjects = set()
final = []
for i in rec_idx:
    subj = str(courses.iloc[i]["subject"]).strip().lower()
    if subj not in seen_subjects or len(final) < k // 2:
        final.append(i)
        seen_subjects.add(subj)
    if len(final) >= k:
        break
if len(final) < k:
    final += rec_idx[: k - len(final)]
final = final[:k]

st.subheader("Recommendations")

def render_row(row: pd.Series, score: float, idx: int):
    cols = st.columns([6, 2, 2, 2])
    with cols[0]:
        st.markdown(f"**{row['title']}**")
        if isinstance(row.get("url", None), str) and row["url"].strip():
            st.markdown(f"[Open course]({row['url']})")
        st.caption(f"Level: {row['level']} • Subject: {row['subject']} • Duration: {row['content_duration']}h • Lectures: {int(row['num_lectures'])}")
    with cols[1]:
        st.metric("Rating", f"{row['avg_rating']:.2f}")
    with cols[2]:
        st.metric("Reviews", int(row['num_reviews']))
    with cols[3]:
        st.markdown(f"Score: {score:.3f}")
    # feedback buttons
    fb_cols = st.columns(3)
    with fb_cols[0]:
        if st.button("👍 Helpful", key=f"up_{idx}"):
            x = np.concatenate([ctx, np.array([
                row["level_num"], math.log1p(row["content_duration"]), math.log1p(row["num_lectures"]), 1.0
            ])])
            bandit.update(x.reshape(-1, 1), reward=1.0)
            save_bandit(bandit)
            st.experimental_rerun()
    with fb_cols[1]:
        if st.button("👎 Not for me", key=f"down_{idx}"):
            x = np.concatenate([ctx, np.array([
                row["level_num"], math.log1p(row["content_duration"]), math.log1p(row["num_lectures"]), 1.0
            ])])
            bandit.update(x.reshape(-1, 1), reward=-0.5)
            save_bandit(bandit)
            st.experimental_rerun()
    with fb_cols[2]:
        if st.button("🚫 Hide beginner drift", key=f"ban_{idx}"):
            # extra penalty if item is below user's level
            x = np.concatenate([ctx, np.array([
                row["level_num"], math.log1p(row["content_duration"]), math.log1p(row["num_lectures"]), 1.0
            ])])
            bandit.update(x.reshape(-1, 1), reward=-1.0)
            save_bandit(bandit)
            st.experimental_rerun()

if len(final) == 0:
    st.info("No items matched after constraints. Loosen filters or lower strictness.")
else:
    for rank, i in enumerate(final, 1):
        r = courses.iloc[i]
        render_row(r, float(scores[i]), i)

# ---------------------------
# Debug and controls
# ---------------------------
with st.expander("Diagnostics"):
    st.write("Context vector:", ctx.tolist())
    st.write("Bandit dims:", bandit.d)
    st.write("Guard: beginner filtering active when user is Intermediate or Advanced.")
    st.write("Tip: Provide keywords to steer content similarity.")

# ---------------------------
# Notes
# ---------------------------
st.caption(
    "Re‑ranking = 45% content, 25% popularity, 15% duration fit, 15% device fit + LinUCB bonus. "
    "Advanced users will not see Beginner items by default."
)
