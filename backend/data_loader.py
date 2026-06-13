"""
Loads and joins all streaming content CSVs into a single DataFrame ready for
content-based recommendation (TF-IDF similarity scoring).

Sources used:
  - data/MoviesOnStreamingPlatforms.csv  → platform availability for movies
  - data/tv_shows.csv                    → platform availability for TV shows
  - data/netflix/netflix_movies_detailed_up_to_2025.csv   → rich movie content
  - data/netflix/netflix_tv_shows_detailed_up_to_2025.csv → rich TV content
  - data/disney/titles.csv + credits.csv → Disney+ content with cast/directors

The Amazon dataset (amazon_prime_movies_tv_2025.csv) is synthetic and excluded
from content enrichment; Prime Video availability is sourced from the platform
base datasets instead.
"""

import ast
import os
import re

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_title(title: str) -> str:
    """Lowercase, strip, remove punctuation – used only for join keys."""
    t = str(title).lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _parse_score(value: str, denom: float) -> float | None:
    """Convert 'X/denom' strings to a float on the 0-10 scale."""
    try:
        num = float(str(value).split("/")[0])
        return round(num * (10.0 / denom), 2)
    except (ValueError, AttributeError):
        return None


def _parse_genre_list(raw) -> str:
    """Convert Python-list-string (Disney format) or plain string to tidy CSV."""
    if pd.isna(raw):
        return ""
    try:
        items = ast.literal_eval(str(raw))
        return ", ".join(str(i).title() for i in items)
    except (ValueError, SyntaxError):
        return str(raw).strip()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_platform_base() -> pd.DataFrame:
    """
    Combine MoviesOnStreamingPlatforms + tv_shows into a unified platform-
    availability table with normalised columns.
    """
    movies = pd.read_csv(os.path.join(DATA_DIR, "MoviesOnStreamingPlatforms.csv"))
    tv = pd.read_csv(os.path.join(DATA_DIR, "tv_shows.csv"))

    movies["content_type"] = "movie"
    tv["content_type"] = "tv"

    # Scores – movies file lacks IMDb; tv file lacks nothing but that's fine
    movies["imdb_score"] = None
    movies["rotten_tomatoes"] = movies["Rotten Tomatoes"].apply(
        lambda x: _parse_score(x, 100)
    )
    tv["imdb_score"] = tv["IMDb"].apply(lambda x: _parse_score(x, 10))
    tv["rotten_tomatoes"] = tv["Rotten Tomatoes"].apply(
        lambda x: _parse_score(x, 100)
    )

    keep = [
        "Title", "Year", "Age", "Netflix", "Hulu", "Prime Video", "Disney+",
        "content_type", "imdb_score", "rotten_tomatoes",
    ]
    base = pd.concat([movies[keep], tv[keep]], ignore_index=True)
    base.columns = [
        "title", "year", "age_rating", "netflix", "hulu", "prime_video",
        "disney_plus", "content_type", "imdb_score", "rotten_tomatoes",
    ]

    # Coerce platform flags to int (already 0/1 in source)
    for col in ("netflix", "hulu", "prime_video", "disney_plus"):
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0).astype(int)

    base["title_key"] = base["title"].apply(_normalize_title)
    base["year"] = pd.to_numeric(base["year"], errors="coerce")
    return base


def _load_netflix_content() -> pd.DataFrame:
    """
    Combine Netflix movies + TV shows; keep columns useful for TF-IDF and scoring.
    """
    movies = pd.read_csv(
        os.path.join(DATA_DIR, "netflix", "netflix_movies_detailed_up_to_2025.csv")
    )
    tv = pd.read_csv(
        os.path.join(DATA_DIR, "netflix", "netflix_tv_shows_detailed_up_to_2025.csv")
    )
    movies["content_type"] = "movie"
    tv["content_type"] = "tv"

    nf = pd.concat([movies, tv], ignore_index=True)
    nf = nf.rename(columns={
        "release_year": "year",
        "vote_average": "tmdb_score",
        "vote_count": "tmdb_votes",
    })

    nf["title_key"] = nf["title"].apply(_normalize_title)
    nf["year"] = pd.to_numeric(nf["year"], errors="coerce")

    # Keep only what enriches the final table
    keep = [
        "title_key", "year", "description", "genres", "cast", "director",
        "language", "country", "tmdb_score", "tmdb_votes", "popularity",
        "content_type",
    ]
    nf = nf[keep].drop_duplicates(subset=["title_key", "year", "content_type"])
    return nf


def _load_disney_content() -> pd.DataFrame:
    """
    Load Disney titles and aggregate cast / directors from the credits file.
    """
    titles = pd.read_csv(os.path.join(DATA_DIR, "disney", "titles.csv"))
    credits = pd.read_csv(os.path.join(DATA_DIR, "disney", "credits.csv"))

    # Top-5 actors per title, all directors
    actors = (
        credits[credits["role"] == "ACTOR"]
        .groupby("id")["name"]
        .apply(lambda s: ", ".join(s.iloc[:5]))
        .reset_index()
        .rename(columns={"name": "cast"})
    )
    directors = (
        credits[credits["role"] == "DIRECTOR"]
        .groupby("id")["name"]
        .apply(lambda s: ", ".join(s))
        .reset_index()
        .rename(columns={"name": "director"})
    )

    titles = titles.merge(actors, on="id", how="left")
    titles = titles.merge(directors, on="id", how="left")

    titles["genres"] = titles["genres"].apply(_parse_genre_list)
    titles["content_type"] = titles["type"].map({"MOVIE": "movie", "SHOW": "tv"})
    titles["title_key"] = titles["title"].apply(_normalize_title)
    titles["year"] = pd.to_numeric(titles["release_year"], errors="coerce")

    keep = [
        "title_key", "year", "description", "genres", "cast", "director",
        "imdb_score", "imdb_votes", "tmdb_popularity", "tmdb_score",
        "age_certification", "content_type",
    ]
    titles = titles[keep].rename(
        columns={
            "imdb_score": "disney_imdb_score",
            "imdb_votes": "disney_imdb_votes",
            "tmdb_popularity": "popularity",
            "age_certification": "disney_age_cert",
        }
    )
    titles = titles.drop_duplicates(subset=["title_key", "year", "content_type"])
    return titles


# ---------------------------------------------------------------------------
# Join logic
# ---------------------------------------------------------------------------

def _merge_on_title_year(
    base: pd.DataFrame, enrichment: pd.DataFrame, suffix: str
) -> pd.DataFrame:
    """
    Merge enrichment data into base on (title_key, year).  Falls back to a
    title_key-only match for rows whose years differ by ≤1 year (release vs.
    platform-added date skew).
    """
    content_cols = [c for c in enrichment.columns if c not in ("title_key", "year")]

    # Exact match
    merged = base.merge(
        enrichment,
        on=["title_key", "year"],
        how="left",
        suffixes=("", f"_{suffix}"),
    )

    # Fill unmatched rows via title_key-only join (pick closest year)
    unmatched_mask = merged[content_cols[0]].isna()
    if unmatched_mask.any():
        fallback = base[unmatched_mask][["title_key", "year"]].merge(
            enrichment, on="title_key", how="left", suffixes=("_base", "")
        )
        fallback = fallback[
            (fallback["year_base"] - fallback["year"]).abs() <= 1
        ].drop_duplicates(subset=["title_key", "year_base"])
        fallback = fallback.drop(columns=["year"]).rename(columns={"year_base": "year"})

        # Patch the unmatched rows
        merged = merged.merge(
            fallback.add_suffix(f"_{suffix}_fb"),
            left_on=["title_key", "year"],
            right_on=[f"title_key_{suffix}_fb", f"year_{suffix}_fb"],
            how="left",
        )
        for col in content_cols:
            fb_col = f"{col}_{suffix}_fb"
            if fb_col in merged.columns:
                merged[col] = merged[col].combine_first(merged[fb_col])
                merged.drop(columns=[fb_col], inplace=True)
        drop_extra = [c for c in merged.columns if c.endswith(f"_{suffix}_fb")]
        merged.drop(columns=drop_extra, inplace=True, errors="ignore")

    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(include_orphans: bool = True) -> pd.DataFrame:
    """
    Build and return the master recommendation DataFrame.

    Parameters
    ----------
    include_orphans : bool
        When True (default), append Netflix- and Disney-sourced titles that
        have no entry in the platform base datasets.  These rows get platform
        flags inferred from their source (netflix=1 or disney_plus=1).

    Returns
    -------
    pd.DataFrame with columns:
        title, year, content_type, age_rating,
        netflix, hulu, prime_video, disney_plus,
        description, genres, cast, director,
        language, country, imdb_score, rotten_tomatoes,
        tmdb_score, tmdb_votes, popularity,
        tfidf_soup
    """
    base = _load_platform_base()
    netflix = _load_netflix_content()
    disney = _load_disney_content()

    # --- Enrich base with Netflix content ---
    df = _merge_on_title_year(base, netflix, "nf")

    # Backfill imdb_score from Netflix tmdb_score where missing
    if "disney_imdb_score" not in df.columns:
        df["disney_imdb_score"] = None

    # --- Enrich base with Disney content ---
    disney_content_cols = [
        c for c in disney.columns if c not in ("title_key", "year", "content_type")
    ]
    for col in disney_content_cols:
        if col not in df.columns:
            df[col] = None

    df = _merge_on_title_year(df, disney, "ds")

    # Consolidate overlapping content columns (prefer Netflix, fill from Disney)
    def _coalesce(df, primary, fallback):
        if primary in df.columns and fallback in df.columns:
            df[primary] = df[primary].combine_first(df[fallback])
            df.drop(columns=[fallback], inplace=True)
        return df

    df = _coalesce(df, "description", "description_ds")
    df = _coalesce(df, "genres", "genres_ds")
    df = _coalesce(df, "cast", "cast_ds")
    df = _coalesce(df, "director", "director_ds")
    df = _coalesce(df, "imdb_score", "disney_imdb_score")
    df = _coalesce(df, "tmdb_score", "tmdb_score_ds")
    df = _coalesce(df, "popularity", "popularity_ds")
    df = _coalesce(df, "age_rating", "disney_age_cert")

    # --- Optionally append orphan titles not in the platform base ---
    if include_orphans:
        base_keys = set(zip(base["title_key"], base["year"]))

        def _orphan_rows(enrichment_df: pd.DataFrame, platform_col: str) -> pd.DataFrame:
            mask = ~enrichment_df.apply(
                lambda r: (r["title_key"], r["year"]) in base_keys, axis=1
            )
            orphans = enrichment_df[mask].copy()
            orphans["title"] = orphans["title_key"]
            for col in ("netflix", "hulu", "prime_video", "disney_plus"):
                orphans[col] = 1 if col == platform_col else 0
            return orphans

        nf_orphans = _orphan_rows(netflix, "netflix")
        ds_orphans = _orphan_rows(disney, "disney_plus")

        df = pd.concat([df, nf_orphans, ds_orphans], ignore_index=True)

    # --- Clean up and derive columns ---

    # Drop helper join key
    df.drop(columns=["title_key"], inplace=True, errors="ignore")

    # Drop any duplicate columns that snuck in from merges
    df = df.loc[:, ~df.columns.duplicated()]

    # Deduplicate on (normalised title, year, content_type) keeping the row
    # with the most non-null values
    df["_title_key"] = df["title"].apply(_normalize_title)
    df["_nulls"] = df.isnull().sum(axis=1)
    df = (
        df.sort_values("_nulls")
        .drop_duplicates(subset=["_title_key", "year", "content_type"], keep="first")
        .drop(columns=["_title_key", "_nulls"])
    )

    # Normalise platform flags to 0/1 int
    for col in ("netflix", "hulu", "prime_video", "disney_plus"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Build TF-IDF soup: description + genres + cast + director
    def _build_soup(row) -> str:
        parts = []
        if pd.notna(row.get("description")) and row["description"]:
            parts.append(str(row["description"]))
        if pd.notna(row.get("genres")) and row["genres"]:
            # Repeat genres to up-weight them
            genres_text = str(row["genres"]).replace(",", " ")
            parts.append(genres_text)
            parts.append(genres_text)
        if pd.notna(row.get("cast")) and row["cast"]:
            parts.append(str(row["cast"]))
        if pd.notna(row.get("director")) and row["director"]:
            parts.append(str(row["director"]))
        return " ".join(parts).strip()

    df["tfidf_soup"] = df.apply(_build_soup, axis=1)

    # Final column ordering
    ordered = [
        "title", "year", "content_type", "age_rating",
        "netflix", "hulu", "prime_video", "disney_plus",
        "description", "genres", "cast", "director",
        "language", "country",
        "imdb_score", "rotten_tomatoes", "tmdb_score", "tmdb_votes", "popularity",
        "tfidf_soup",
    ]
    present = [c for c in ordered if c in df.columns]
    extra = [c for c in df.columns if c not in ordered]
    df = df[present + extra].reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = load_data()
    print(f"Final shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nPlatform counts:")
    for col in ("netflix", "hulu", "prime_video", "disney_plus"):
        print(f"  {col}: {df[col].sum()}")
    print(f"\nContent type breakdown:\n{df['content_type'].value_counts()}")
    print(f"\nTF-IDF soup sample:\n{df['tfidf_soup'].iloc[0][:300]}")
    print(f"\nNull counts in key columns:")
    key_cols = ["description", "genres", "cast", "director", "tfidf_soup"]
    print(df[key_cols].isnull().sum())
