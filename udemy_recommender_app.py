"""
Simple Context-Aware Udemy Recommender (dataset-only features)

Scoring = content similarity (embeddings or TF-IDF fallback)
        + popularity (rating/reviews/subscribers, rating ignored if absent)
        + duration fit vs hours/week
Gating = block easier courses for higher-skill users, with a toggle
Filters = optional keyword match
"""

import os
import re
import math
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="Simple Udemy Recommender", layout="wide")
st.title("Simple Udemy Course Recommender")

DATA_CSV = "udemy_courses_cleaned.csv"  # fallback to udemy_courses.csv
EMB_MODULE = "udemy_course_embeddings"   # optional: get_course_embeddings(df), get_query_embedding(text)

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

    # Ensure expected columns exist
    expected = [
        "course_id", "title", "url", "price", "num_subscribers",
        "avg_rating", "num_reviews", "num_lectures", "level",
        "content_duration", "published_timestamp", "subject", "description"
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan

    # Clean types
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

    # Difficulty mapping
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
    present = [c for c in base_text_cols if c in df.columns]
    if present:
        df[present] = df[present].astype(str).replace("nan", "", regex=False)
        df["text"] = df[present].agg(" \n".join, axis=1)
    else:
        df["text"] = ""

    # Fill missing titles to avoid 'nan'
    missing_title = df["title"].isna() | (df["title"].astype(str).str.strip() == "")
    fallback_ids = (df["course_id"].fillna(0).astype(int).astype(str)
                    if "course_id" in df.columns else
                    pd.Series(range(len(df))).astype(str))
    df.loc[missing_title, "title"] = "Course #" + fallback_ids[missing_title]

    return df

courses = load_courses()

# ---------------------------
# Embeddings
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    try:
        imp = __import__(EMB_MODULE)
        return imp  # expects get_course_embeddings(df), get_query_embedding(text)
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
# Scoring components
# ---------------------------
@st.cache_data(show_spinner=False)
def popularity_scores(df: pd.DataFrame) -> np.ndarray:
    X = pd.DataFrame({
        "avg_rating": df["avg_rating"],
        "num_reviews": df["num_reviews"],
        "num_subscribers": df["num_subscribers"],
    })
    # If ratings missing for all rows, drop rating from score
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
        return 0.6 * Xn[:, 1] + 0.4 * Xn[:, 2]
    return rating_weight * Xn[:, 0] + 0.35 * Xn[:, 1] + 0.25 * Xn[:, 2]

POP = popularity_scores(courses)

def duration_fit(df: pd.DataFrame, hours_per_week: int, horizon_weeks: int = 6) -> np.ndarray:
    cap = max(1, hours_per_week * horizon_weeks)
    dur = df["content_duration"].fillna(0.0).to_numpy()
    return np.where(dur <= cap, 1.0, np.clip(cap / (dur + 1e-6), 0.1, 1.0))

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
    user_skill: str,
    strict_gate: bool,
    weights=None,
) -> np.ndarray:
    # Upweight content when ratings are missing
    if weights is None:
        w = 0.6 if pd.isna(df["avg_rating"]).all() else 0.5
        weights = (w, 0.35, 0.15)  # content, popularity, duration
    a, b, c = weights

    sim = cosine_similarity(query_vec, COURSE_EMB).ravel()
    if sim.max() - sim.min() > 1e-9:
        sim = (sim - sim.min()) / (sim.max() - sim.min())

    pop = POP
    dur = duration_fit(df, hours_per_week)
    base = a * sim + b * pop + c * dur

    guard = difficulty_guard(df, user_skill, strict=strict_gate)
    return base * guard

# ---------------------------
# UI
# ---------------------------
with st.sidebar:
    st.header("Context")
    skill = st.selectbox("Current skill level", ["Beginner", "Intermediate", "Advanced"], index=1)
    hours = st.slider("Time availability (hours/week)", 1, 40, 6)
    query = st.text_input("Interests/keywords", value="python data analysis pandas")
    require_match = st.checkbox("Require keyword match", value=True)
    k = st.slider("How many recommendations", 5, 30, 10)
    strict_gate = st.checkbox("Strict difficulty gating", value=True)

# Query embedding
qv = embedder.get_query_embedding(query)
if not isinstance(qv, np.ndarray):
    try:
        qv = qv.toarray()
    except Exception:
        qv = np.asarray(qv)
qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-8)

# Score and rank
scores = hybrid_score(courses, qv, hours, skill, strict_gate)
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

# Keep valid titles only
mask_ok = courses.iloc[rec_idx]["title"].astype(str).str.strip().ne("")
rec_idx = list(np.array(rec_idx)[np.where(mask_ok)[0]])

# Simple diversity by subject
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

# ---------------------------
# Render
# ---------------------------
st.subheader("Recommendations")

def safe_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default

if len(final) == 0:
    st.warning("No items matched. Try turning off strict gating or relax keyword match.")
else:
    for rank, i in enumerate(final, 1):
        r = courses.iloc[i]
        cols = st.columns([6, 2, 2, 2])
        with cols[0]:
            ttl = str(r["title"]).strip() or f"Course #{safe_int(r.get('course_id', i), i)}"
            st.markdown(f"**{ttl}**")
            url = str(r.get("url", "")).strip()
            if url:
                st.markdown(f"[Open course]({url})")
            st.caption(
                f"Level: {r['level']} • Subject: {r['subject']} • "
                f"Duration: {r['content_duration']}h • Lectures: {safe_int(r['num_lectures'])}"
            )
        with cols[1]:
            rating_val = r.get("avg_rating")
            if pd.isna(rating_val) or float(rating_val) == 0.0:
                st.metric("Rating", "—")
            else:
                st.metric("Rating", f"{float(rating_val):.2f}")
        with cols[2]:
            st.metric("Reviews", safe_int(r.get("num_reviews", 0)))
        with cols[3]:
            st.markdown(f"Score: {scores[i]:.3f}")

# ---------------------------
# Diagnostics
# ---------------------------
with st.expander("Diagnostics"):
    level_counts = (
        courses["level"]
        .dropna()
        .astype(str)
        .str.lower()
        .value_counts()
        .head(10)              # limit here
        .to_dict()
    )
    st.write({
        "rows": len(courses),
        "non_null_titles": int(courses["title"].astype(str).str.strip().ne("").sum()),
        "levels": level_counts,
        "ratings_missing_all": bool(courses["avg_rating"].isna().all()),
    })
