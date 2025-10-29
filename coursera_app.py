import os
import math
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Context-Aware Course Recommender", page_icon="🎯", layout="wide")

# ---------- Data loading ----------
def parse_skills(x):
    if pd.isna(x): return []
    s = str(x).strip()
    if s.startswith("{") and s.endswith("}"): s = s[1:-1]
    parts = [p.strip().strip('"').strip("'") for p in s.split(",")]
    return [p for p in parts if p]

def parse_reviews(x):
    s = str(x).lower().strip()
    try:
        if s.endswith("k"): return float(s[:-1]) * 1_000
        if s.endswith("m"): return float(s[:-1]) * 1_000_000
        return float(s)
    except: return np.nan

def duration_to_months(s):
    s = str(s)
    if "1 - 3" in s: return 2.0
    if "3 - 6" in s: return 4.5
    if "6 - 12" in s: return 9.0
    if "< 1" in s or "Less" in s: return 0.5
    return 4.0

@st.cache_data
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["skills_list"] = df["skills"].apply(parse_skills)
    df["reviewcount_num"] = df["reviewcount"].apply(parse_reviews).fillna(0)
    df["duration_months"] = df["duration"].apply(duration_to_months)
    df["level"] = df["level"].fillna("Unknown").str.strip()
    df["certificatetype"] = df["certificatetype"].fillna("Unknown").str.strip()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(df["rating"].median())
    df["text_blob"] = (
        df["partner"].fillna("") + " " +
        df["course"].fillna("") + " " +
        df["skills_list"].apply(lambda xs: " ".join(xs))
    ).str.lower()
    return df

# ---------- Features ----------
@st.cache_resource
def build_features(df: pd.DataFrame):
    skills_corpus = df["skills_list"].apply(lambda xs: ", ".join(xs))
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
    X = tfidf.fit_transform(skills_corpus)

    rating = df["rating"].values
    pop = np.log1p(df["reviewcount_num"].values)
    r_norm = (rating - rating.min()) / (rating.max() - rating.min() + 1e-9)
    p_norm = (pop - pop.min()) / (pop.max() - pop.min() + 1e-9)
    popularity = 0.6*r_norm + 0.4*p_norm
    return tfidf, X, popularity

LEVELS = ["Beginner","Intermediate","Advanced","Unknown"]
# --- stronger context ---
def context_score(df, cand_idx, level, hours_per_week, device, study_time):
    lvl_match = (df.loc[cand_idx, "level"].str.lower() == level.lower()).astype(float).values

    desired_months = np.clip(hours_per_week/4.0, 0.5, 9)  # tighter mapping
    dur = df.loc[cand_idx, "duration_months"].values
    dur_fit = np.exp(-((dur - desired_months)**2) / (2*(1.0**2)))  # narrower peak

    dev_pen = 0.0
    if device == "mobile":
        # penalize long programs more aggressively on mobile
        dur_norm = (dur - dur.min()) / (dur.max() - dur.min() + 1e-9)
        dev_pen = 0.15 * dur_norm

    # study_time still neutral without logs; keep tiny jitter to break ties
    jitter = 1e-6 * np.random.randn(len(dur))

    # weights: duration fit dominates context, then level, then device penalty
    ctx = 0.6*dur_fit + 0.3*lvl_match - dev_pen + jitter
    return np.clip(ctx, 0, 1)

def recommend(df, tfidf, X, popularity, query_skills, level, hours_per_week, device, study_time, top_k, level_filters, min_rating):
    mask = df["rating"] >= float(min_rating)
    if level_filters: mask &= df["level"].isin(level_filters)
    cand_idx = np.where(mask.values)[0]
    if cand_idx.size == 0: return pd.DataFrame()

    if query_skills.strip():
        q_vec = tfidf.transform([query_skills.lower()])
        sim = cosine_similarity(q_vec, X[cand_idx]).ravel()
    else:
        sim = np.zeros(len(cand_idx))

    # normalize sim and popularity per candidate set
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-9)
    pop_c = popularity[cand_idx]
    pop_c = (pop_c - pop_c.min()) / (pop_c.max() - pop_c.min() + 1e-9)

    ctx = context_score(df, cand_idx, level, hours_per_week, device, study_time)

    # stronger context blend
    score = 0.45*sim + 0.20*pop_c + 0.35*ctx

    order_local = np.argsort(-score)[:top_k]
    order = cand_idx[order_local]
    out = df.loc[order, ["partner","course","level","rating","reviewcount_num","duration_months","certificatetype","skills_list"]].copy()
    out.insert(0, "score", score[order_local])
    out.insert(1, "ctx", ctx[order_local])
    out.insert(2, "sim", sim[order_local])
    out.insert(3, "pop", pop_c[order_local])
    return out.reset_index(drop=True)

# ---------- UI: data source ----------
st.title("Context-Aware Course Recommender")
st.caption("Content-based core with context reranking")

uploader = st.file_uploader("Upload Coursera.csv", type=["csv"])
if uploader:
    df = load_df(uploader)
elif os.path.exists("Coursera.csv"):
    df = load_df("Coursera.csv")
else:
    st.info("Upload Coursera.csv to continue.")
    st.stop()

tfidf, X, popularity = build_features(df)

# ---------- Sidebar: context and filters ----------
with st.sidebar:
    st.header("Context")
    hours = st.slider("Time availability (hours/week)", 1, 30, 6)
    device = st.radio("Device", ["desktop","mobile"], horizontal=True, index=0)
    study_time = st.radio("Preferred study time", ["morning","evening","any"], horizontal=True, index=2)

    st.header("Filters")
    query = st.text_input("Skills or keywords", placeholder="e.g., python, sql, data analysis")
    min_rating = st.slider("Min rating", 0.0, 5.0, 4.5, 0.1)
    level_filters = st.multiselect("Levels", LEVELS, default=[])
    top_k = st.slider("Top N", 5, 20, 10)

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["🔎 Recommendations", "📊 EDA"])

with tab1:
    res = recommend(df, tfidf, X, popularity, query, "Unknown", hours, device, study_time, top_k, level_filters, min_rating)
    if res.empty:
        st.warning("No matches for current filters.")
    else:
        st.subheader("Recommended for you")
        st.caption(
            """
            **Score meaning**  
            • **score** → Final weighted value combining similarity, popularity, and context.  
            • **sim** → Skill and keyword match strength.  
            • **pop** → Popularity based on ratings and review count.  
            • **ctx** → Fit with your current situation (time, device).  
            """
        )
        st.dataframe(res, use_container_width=True)

        # Cards view
        st.divider()
        st.subheader("Cards")
        for i, row in res.iterrows():
            with st.container(border=True):
                c1, c2 = st.columns([5, 2])
                with c1:
                    st.markdown(f"**{row['course']}**  \n{row['partner']} • {row['level']} • {row['certificatetype']}")
                    st.markdown(
                        f"Rating **{row['rating']:.1f}** · Reviews **{int(row['reviewcount_num']):,}** · Duration ~**{row['duration_months']:.1f} mo**"
                    )
                    skills = ", ".join(row["skills_list"][:12])
                    if skills:
                        st.caption(f"Skills: {skills}")
                with c2:
                    st.metric("Final score", f"{row['score']:.2f}")
                    st.text(f"ctx={row['ctx']:.2f}")
                    st.text(f"sim={row['sim']:.2f}")
                    st.text(f"pop={row['pop']:.2f}")

with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Rating distribution")
        hist = alt.Chart(df).mark_bar().encode(
            x=alt.X("rating:Q", bin=alt.Bin(maxbins=20), title="Rating"),
            y=alt.Y("count()", title="Count")
        ).properties(height=250)
        st.altair_chart(hist, use_container_width=True)

        st.subheader("Levels")
        levels = df["level"].value_counts().reset_index()
        levels.columns = ["level","count"]
        bar = alt.Chart(levels).mark_bar().encode(x="level:N", y="count:Q")
        st.altair_chart(bar, use_container_width=True)

    with c2:
        st.subheader("Reviews (clipped at 98th pct)")
        clipped = df.copy()
        cap = clipped["reviewcount_num"].quantile(0.98)
        clipped["reviews_clipped"] = np.minimum(clipped["reviewcount_num"], cap)
        rev_hist = alt.Chart(clipped).mark_bar().encode(
            x=alt.X("reviews_clipped:Q", bin=alt.Bin(maxbins=25), title="Reviews"),
            y="count()"
        ).properties(height=250)
        st.altair_chart(rev_hist, use_container_width=True)

        st.subheader("Top skills")
        skills_flat = [s.strip() for lst in df["skills_list"] for s in lst if s.strip()]
        top_sk = Counter(skills_flat).most_common(20)
        sk_df = pd.DataFrame(top_sk, columns=["skill","count"])
        sk_bar = alt.Chart(sk_df).mark_bar().encode(x="count:Q", y=alt.Y("skill:N", sort="-x"))
        st.altair_chart(sk_bar, use_container_width=True)

# ---------- Notes ----------
with st.expander("Design notes"):
    st.markdown(
        """
- Item features: TF-IDF on skills, rating, reviews, duration, level, certificate type.
- Base rank: cosine similarity with optional query; blended with popularity.
- Context rerank: level match, duration fit from hours/week, light penalty for mobile + long duration.
- Cold start: falls back to popularity when query is empty.
- Extend next: log feedback, train logistic reranker on [context ⊗ item] features, add titles/partner features.
"""
    )
