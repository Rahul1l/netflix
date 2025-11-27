import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try streamlit-local-storage (optional)
try:
    from streamlit_local_storage import LocalStorage
    HAS_LOCAL_STORAGE = True
except Exception:
    HAS_LOCAL_STORAGE = False
    LocalStorage = None

# --------------------
# Page config
# --------------------
st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Netflix Recommendation System & Analytics Dashboard")
st.markdown(
    """
Upload a **Netflix-style CSV** and this app will:

1. **Clean** messy data (whitespace, missing values, basic type fixes)  
2. Show a **KPI dashboard** with interactive visualizations  
3. Offer **content-based recommendations** using genres + description  
4. (Optional) Remember your **last selected title** using local storage
"""
)

# Initialize local storage if available
localS = LocalStorage() if HAS_LOCAL_STORAGE else None


# --------------------
# Data cleaning
# --------------------
def clean_netflix_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Strip whitespace from string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"nan": np.nan, "None": np.nan})
        )

    # Parse date_added
    if "date_added" in df.columns:
        df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")

    # Ensure numeric release_year
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")

    # Parse duration into a numeric column (minutes or seasons count)
    if "duration" in df.columns:
        def parse_duration(x):
            x = str(x)
            parts = x.split()
            if not parts:
                return np.nan
            num = parts[0]
            if not num.isdigit():
                return np.nan
            return int(num)

        df["duration_value"] = df["duration"].apply(parse_duration)

    # Drop exact duplicates
    df = df.drop_duplicates()

    # Fill some important categorical columns
    for col in ["country", "rating", "listed_in"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Clean title and type
    for col in ["title", "type"]:
        if col in df.columns:
            df[col] = df[col].str.title()

    return df


# --------------------
# KPI + charts
# --------------------
def show_kpis_and_charts(df: pd.DataFrame):
    st.subheader("📊 Key Performance Indicators (KPIs)")

    total_titles = len(df)
    num_movies = None
    num_tvshows = None

    if "type" in df.columns:
        num_movies = (df["type"] == "Movie").sum()
        num_tvshows = (df["type"] == "Tv Show").sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Titles", f"{total_titles:,}")
    if num_movies is not None:
        c2.metric("Movies", f"{num_movies:,}")
    if num_tvshows is not None:
        c3.metric("TV Shows", f"{num_tvshows:,}")

    st.markdown("---")
    st.subheader("📈 Visualizations")

    col_left, col_right = st.columns(2)

    # ---- Titles by Type ----
    if "type" in df.columns:
        with col_left:
            st.markdown("**Titles by Type**")
            type_counts = (
                df["type"]
                .value_counts()
                .reset_index()
                .rename(columns={"index": "type", "type": "count"})
            )
            # Ensure no duplicate columns
            type_counts = type_counts.loc[:, ~type_counts.columns.duplicated()]

            chart = (
                alt.Chart(type_counts)
                .mark_bar()
                .encode(
                    x=alt.X("type:N", title="Type"),
                    y=alt.Y("count:Q", title="Number of Titles"),
                    tooltip=["type", "count"],
                )
            )
            st.altair_chart(chart, use_container_width=True)

    # ---- Top Genres ----
    if "listed_in" in df.columns:
        with col_right:
            st.markdown("**Top Genres**")

            # Split multi-genre values into rows
            genre_series = (
                df["listed_in"]
                .dropna()
                .str.split(",", expand=True)
                .stack()
                .str.strip()
                .rename("genre")
            )

            genre_counts = (
                genre_series.to_frame()
                .groupby("genre")
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .head(15)
            )
            genre_counts = genre_counts.loc[:, ~genre_counts.columns.duplicated()]

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
    col_left2, col_right2 = st.columns(2)

    # ---- Titles by Release Year ----
    if "release_year" in df.columns:
        with col_left2:
            st.markdown("**Titles by Release Year**")
            year_counts = (
                df.dropna(subset=["release_year"])
                .groupby("release_year")
                .size()
                .reset_index(name="count")
                .sort_values("release_year")
            )
            year_counts = year_counts.loc[:, ~year_counts.columns.duplicated()]

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

    # ---- Top Countries ----
    if "country" in df.columns:
        with col_right2:
            st.markdown("**Top Countries**")
            country_series = (
                df["country"]
                .fillna("Unknown")
                .str.split(",", expand=True)
                .stack()
                .str.strip()
                .rename("country")
            )

            country_counts = (
                country_series.to_frame()
                .groupby("country")
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .head(10)
            )
            country_counts = country_counts.loc[:, ~country_counts.columns.duplicated()]

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
# Recommender
# --------------------
@st.cache_resource(show_spinner=False)
def build_recommender_model(df: pd.DataFrame):
    df = df.copy()
    if "title" not in df.columns:
        return None, None

    text_components = []
    if "listed_in" in df.columns:
        text_components.append(df["listed_in"].fillna(""))
    if "description" in df.columns:
        text_components.append(df["description"].fillna(""))

    if not text_components:
        return None, None

    if len(text_components) == 1:
        combined = text_components[0]
    else:
        combined = text_components[0] + " " + text_components[1]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(combined)

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim, df.reset_index(drop=True)


def get_recommendations(title, cosine_sim, df, n_recs=10):
    if cosine_sim is None or df is None:
        return pd.DataFrame()

    titles = df["title"].astype(str)
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
    columns_to_show = [c for c in ["title", "type", "listed_in", "country", "similarity"] if c in recs.columns]
    return recs[columns_to_show]


# --------------------
# Sidebar: file upload
# --------------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload Netflix CSV",
    type=["csv"],
    help="Upload a CSV with columns like: show_id, type, title, country, date_added, release_year, rating, duration, listed_in, description"
)

st.sidebar.markdown(
    """
**Recommended columns:**

- `show_id`, `type`, `title`, `director`, `cast`  
- `country`, `date_added`, `release_year`  
- `rating`, `duration`, `listed_in`, `description`
"""
)


# --------------------
# Main logic
# --------------------
if uploaded_file is None:
    st.info("👆 Upload a CSV file from the sidebar to start.")
    st.markdown(
        """
You can use your own Netflix catalogue or a synthetic dataset  
with ~1000 rows that follows a similar schema.
"""
    )
else:
    # Raw preview
    raw_df = pd.read_csv(uploaded_file)
    st.subheader("🧾 Raw Data Preview")
    st.dataframe(raw_df.head())
    st.caption(f"Raw shape: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")

    # Cleaning
    with st.spinner("Cleaning data..."):
        df = clean_netflix_data(raw_df)

    st.subheader("🧹 Cleaned Data Preview")
    st.dataframe(df.head())
    st.caption(f"Cleaned shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # KPIs & Visuals
    show_kpis_and_charts(df)

    st.markdown("---")
    st.subheader("🎯 Content-Based Recommendations")

    if "title" not in df.columns or df["title"].dropna().empty:
        st.warning("No valid `title` column found. Cannot build recommendations.")
    else:
        titles_sorted = sorted(df["title"].dropna().unique().tolist())

        # Restore last selected title from local storage if available
        default_index = 0
        if HAS_LOCAL_STORAGE and localS is not None:
            try:
                saved_title = localS.getItem("selected_title")
                if saved_title and saved_title in titles_sorted:
                    default_index = titles_sorted.index(saved_title)
            except Exception:
                saved_title = None

        selected_title = st.selectbox(
            "Select a title to get similar recommendations:",
            options=titles_sorted,
            index=default_index if default_index < len(titles_sorted) else 0
        )

        # Save selection to local storage
        if HAS_LOCAL_STORAGE and localS is not None:
            try:
                localS.setItem("selected_title", selected_title)
            except Exception:
                pass

        cosine_sim, model_df = build_recommender_model(df)

        if st.button("🔍 Get Recommendations"):
            with st.spinner("Finding similar titles..."):
                recs = get_recommendations(selected_title, cosine_sim, model_df)

            if recs.empty:
                st.warning("No recommendations could be generated. Check that `listed_in` and/or `description` exist.")
            else:
                st.success(f"Titles similar to **{selected_title}**:")
                st.dataframe(recs)

    # Optional: debug local storage
    if HAS_LOCAL_STORAGE and localS is not None:
        with st.expander("🧪 Local Storage (debug)"):
            try:
                st.json(localS.getAll())
            except Exception:
                st.write("Could not read local storage.")
