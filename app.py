import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from streamlit_local_storage import LocalStorage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# --------------------
# Page config
# --------------------
st.set_page_config(
    page_title="Netflix Recommendation & Insights",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Netflix Recommendation System & Dashboard")
st.markdown(
    """
Upload a **Netflix-style catalogue CSV** and the app will:

1. Clean messy data (whitespaces, missing values, duplicates, types)  
2. Show a **KPI dashboard & visualizations**  
3. Provide **content-based recommendations**  
4. Use **Streamlit Local Storage** to remember your last selected title
"""
)

localS = LocalStorage()  # for browser local storage

# --------------------
# Data cleaning utils
# --------------------
def clean_netflix_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Strip whitespace from object columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan})

    # Parse date_added if present
    if "date_added" in df.columns:
        df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")

    # Convert release_year to numeric
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")

    # Basic duration parsing
    if "duration" in df.columns:
        def parse_duration(x):
            x = str(x)
            if "Season" in x:
                # TV shows: keep seasons count
                num = x.split()[0]
                return int(num) if num.isdigit() else np.nan
            elif "min" in x:
                num = x.split()[0]
                return int(num) if num.isdigit() else np.nan
            return np.nan

        df["duration_parsed"] = df["duration"].apply(parse_duration)

    # Drop exact duplicates
    df = df.drop_duplicates()

    # Fill some common missing columns
    for col in ["country", "rating", "listed_in"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Title/type basic cleaning
    for col in ["title", "type"]:
        if col in df.columns:
            df[col] = df[col].str.title()

    return df


# --------------------
# KPI & visualization
# --------------------
def show_kpis_and_charts(df: pd.DataFrame):
    st.subheader("📊 Key Performance Indicators (KPIs)")

    total_titles = len(df)
    num_movies = len(df[df["type"] == "Movie"]) if "type" in df.columns else np.nan
    num_tvshows = len(df[df["type"] == "Tv Show"]) if "type" in df.columns else np.nan

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Titles", f"{total_titles:,}")
    if not np.isnan(num_movies):
        col2.metric("Movies", f"{num_movies:,}")
    if not np.isnan(num_tvshows):
        col3.metric("TV Shows", f"{num_tvshows:,}")

    st.markdown("---")
    st.subheader("📈 Visualizations")

    c1, c2 = st.columns(2)

    # Distribution by type
    if "type" in df.columns:
        with c1:
            st.markdown("**Titles by Type**")
            chart_df = (
                df["type"]
                .value_counts()
                .reset_index()
                .rename(columns={"index": "type", "type": "count"})
            )
            chart = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X("type:N", title="Type"),
                    y=alt.Y("count:Q", title="Number of Titles"),
                    tooltip=["type", "count"],
                )
            )
            st.altair_chart(chart, use_container_width=True)

    # Top genres / listed_in
    if "listed_in" in df.columns:
        with c2:
            st.markdown("**Top Genres**")
            genre_series = (
                df["listed_in"]
                .dropna()
                .str.split(",", expand=True)
                .stack()
                .str.strip()
            )
            genre_counts = (
                genre_series.value_counts()
                .reset_index()
                .rename(columns={"index": "genre", "listed_in": "count"})
                .head(15)
            )
            chart = (
                alt.Chart(genre_counts)
                .mark_bar()
                .encode(
                    x=alt.X("count:Q", title="Number of Titles"),
                    y=alt.Y("genre:N", sort="-x", title="Genre"),
                    tooltip=["genre", "count"],
                )
            )
            st.altair_chart(chart, use_container_width=True)

    st.markdown("---")
    c3, c4 = st.columns(2)

    # Titles over years
    if "release_year" in df.columns:
        with c3:
            st.markdown("**Titles by Release Year**")
            year_counts = (
                df.dropna(subset=["release_year"])
                .groupby("release_year")
                .size()
                .reset_index(name="count")
            )
            chart = (
                alt.Chart(year_counts)
                .mark_line(point=True)
                .encode(
                    x=alt.X("release_year:O", title="Release Year"),
                    y=alt.Y("count:Q", title="Number of Titles"),
                    tooltip=["release_year", "count"],
                )
            )
            st.altair_chart(chart, use_container_width=True)

    # Top countries
    if "country" in df.columns:
        with c4:
            st.markdown("**Top Countries**")
            country_counts = (
                df["country"]
                .fillna("Unknown")
                .str.split(",", expand=True)
                .stack()
                .str.strip()
                .value_counts()
                .reset_index()
                .rename(columns={"index": "country", "country": "count"})
                .head(10)
            )
            chart = (
                alt.Chart(country_counts)
                .mark_bar()
                .encode(
                    x=alt.X("count:Q", title="Number of Titles"),
                    y=alt.Y("country:N", sort="-x", title="Country"),
                    tooltip=["country", "count"],
                )
            )
            st.altair_chart(chart, use_container_width=True)


# --------------------
# Simple content-based recommender
# --------------------
@st.cache_resource(show_spinner=False)
def build_recommender_model(df: pd.DataFrame):
    df = df.copy()
    if "title" not in df.columns:
        return None, None

    text_parts = []
    if "listed_in" in df.columns:
        text_parts.append(df["listed_in"].fillna(""))
    if "description" in df.columns:
        text_parts.append(df["description"].fillna(""))

    if not text_parts:
        return None, None

    combined_text = (
        text_parts[0] if len(text_parts) == 1 else text_parts[0] + " " + text_parts[1]
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(combined_text)

    # Cosine similarity matrix (can be large, but fine for ~1k rows)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim, df.reset_index(drop=True)


def get_recommendations(title, cosine_sim, df, n_recs=10):
    if cosine_sim is None or df is None:
        return pd.DataFrame()

    titles = df["title"]
    indices = pd.Series(df.index, index=titles.str.lower())

    key = title.lower()
    if key not in indices:
        return pd.DataFrame()

    idx = indices[key]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1 : n_recs + 1]

    movie_indices = [i[0] for i in sim_scores]
    recs = df.iloc[movie_indices].copy()
    recs["similarity"] = [s[1] for s in sim_scores]
    return recs[["title", "type", "listed_in", "country", "similarity"]]


# --------------------
# Sidebar: file upload
# --------------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload Netflix CSV", type=["csv"], help="Upload a CSV with Netflix-style columns"
)

st.sidebar.markdown(
    """
**Expected columns (recommended):**

- `show_id`, `type`, `title`, `director`, `cast`,  
- `country`, `date_added`, `release_year`, `rating`,  
- `duration`, `listed_in`, `description`
"""
)

# --------------------
# Main logic
# --------------------
if uploaded_file is None:
    st.info("👆 Upload a CSV file from the sidebar to get started.")
    st.markdown(
        """
If you don't have a dataset, you can start with the **sample 1000-row dataset** I generated for you.
"""
    )
    st.code("netflix_sample_1000.csv  # place this in your project folder", language="bash")
else:
    # Read & clean
    raw_df = pd.read_csv(uploaded_file)
    st.subheader("🧹 Raw Data Preview")
    st.dataframe(raw_df.head())

    with st.spinner("Cleaning data..."):
        df = clean_netflix_data(raw_df)

    st.subheader("✅ Cleaned Data Preview")
    st.dataframe(df.head())

    # Display shape
    st.caption(f"Dataset shape after cleaning: {df.shape[0]} rows × {df.shape[1]} columns")

    # KPIs & Visuals
    show_kpis_and_charts(df)

    # --------------------
    # Recommendation section
    # --------------------
    st.markdown("---")
    st.subheader("🎯 Content-Based Recommendations")

    # Use local storage to remember last selected title
    try:
        saved_title = localS.getItem("selected_title")
    except Exception:
        saved_title = None

    available_titles = sorted(df["title"].dropna().unique()) if "title" in df.columns else []

    default_index = 0
    if saved_title and saved_title in available_titles:
        default_index = available_titles.index(saved_title)

    if available_titles:
        selected_title = st.selectbox(
            "Select a title to get similar recommendations:",
            options=available_titles,
            index=default_index if default_index < len(available_titles) else 0,
        )

        # Save selection to local storage
        try:
            localS.setItem("selected_title", selected_title)
        except Exception:
            pass  # component might fail silently in some environments

        cosine_sim, model_df = build_recommender_model(df)

        if st.button("Get Recommendations"):
            with st.spinner("Computing recommendations..."):
                recs = get_recommendations(selected_title, cosine_sim, model_df)

            if not recs.empty:
                st.success(f"Recommendations similar to **{selected_title}**:")
                st.dataframe(recs)
            else:
                st.warning("No recommendations available. Check if required columns exist (e.g., `listed_in`, `description`).")
    else:
        st.warning("No `title` column found or it is empty – cannot build recommendations.")

    # --------------------
    # Debug / Local Storage Viewer
    # --------------------
    with st.expander("🧪 Local Storage (Debug)"):
        st.write("Items stored in browser local storage:")
        try:
            all_items = localS.getAll()
            st.json(all_items)
        except Exception:
            st.write("Local storage not available or component not working in this environment.")
