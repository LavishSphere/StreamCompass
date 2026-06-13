"""
Loads and joins all streaming content CSVs into a single DataFrame ready for
content-based recommendation (TF-IDF similarity scoring).

Sources used:
  - data/MoviesOnStreamingPlatforms.csv  → platform availability for movies
  - data/tv_shows.csv                    → platform availability for TV shows
  - data/netflix/netflix_movies_detailed_up_to_2025.csv   → rich movie content
  - data/netflix/netflix_tv_shows_detailed_up_to_2025.csv → rich TV content
  - data/disney/titles.csv + credits.csv → Disney+ content with cast/directors

The Amazon dataset (amazon_prime_movies_tv_2025.csv) contains synthetic/fake
data (fabricated directors, number-appended titles) and is excluded; Prime
Video availability comes from MoviesOnStreamingPlatforms and tv_shows instead.
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
    """Lowercase, strip, remove punctuation – used only as a join key."""
    t = str(title).lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _parse_score(value, denom: float):
    try:
        result = float(str(value).split("/")[0]) * (10.0 / denom)
        return None if (result != result) else round(result, 2)  # NaN check
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


def _coalesce_merge(left: pd.DataFrame, right: pd.DataFrame, on: list) -> pd.DataFrame:
    """
    Left-merge `right` into `left` on `on`.  For every column in `right` that
    also exists in `left`, the merged value fills NaNs in `left`; the temporary
    `_right` column is then dropped.  New columns are kept as-is.
    """
    result = left.merge(right, on=on, how="left", suffixes=("", "_right"))
    right_new_cols = [c for c in right.columns if c not in on]
    for col in right_new_cols:
        r_col = f"{col}_right"
        if r_col in result.columns:
            result[col] = result[col].combine_first(result[r_col])
            result.drop(columns=[r_col], inplace=True)
    return result


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------


def _load_platform_base() -> pd.DataFrame:
    """
    Combine MoviesOnStreamingPlatforms + tv_shows into a unified platform-
    availability table.
    """
    movies = pd.read_csv(os.path.join(DATA_DIR, "MoviesOnStreamingPlatforms.csv"))
    tv = pd.read_csv(os.path.join(DATA_DIR, "tv_shows.csv"))

    movies["content_type"] = "movie"
    tv["content_type"] = "tv"

    movies["imdb_score"] = None
    movies["rotten_tomatoes"] = movies["Rotten Tomatoes"].apply(lambda x: _parse_score(x, 100))
    tv["imdb_score"] = tv["IMDb"].apply(lambda x: _parse_score(x, 10))
    tv["rotten_tomatoes"] = tv["Rotten Tomatoes"].apply(lambda x: _parse_score(x, 100))

    keep = [
        "Title",
        "Year",
        "Age",
        "Netflix",
        "Hulu",
        "Prime Video",
        "Disney+",
        "content_type",
        "imdb_score",
        "rotten_tomatoes",
    ]
    base = pd.concat([movies[keep], tv[keep]], ignore_index=True)
    base.columns = [
        "title",
        "year",
        "age_rating",
        "netflix",
        "hulu",
        "prime_video",
        "disney_plus",
        "content_type",
        "imdb_score",
        "rotten_tomatoes",
    ]

    for col in ("netflix", "hulu", "prime_video", "disney_plus"):
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0).astype(int)

    base["year"] = pd.to_numeric(base["year"], errors="coerce")
    base["title_key"] = base["title"].apply(_normalize_title)
    return base


def _load_netflix_content() -> pd.DataFrame:
    """
    Combine Netflix movies + TV shows; keep columns useful for TF-IDF and scoring.
    """
    movies = pd.read_csv(
        os.path.join(DATA_DIR, "netflix", "netflix_movies_detailed_up_to_2025.csv")
    )
    tv = pd.read_csv(os.path.join(DATA_DIR, "netflix", "netflix_tv_shows_detailed_up_to_2025.csv"))
    movies["content_type"] = "movie"
    tv["content_type"] = "tv"

    nf = pd.concat([movies, tv], ignore_index=True)
    nf = nf.rename(
        columns={
            "release_year": "year",
            "vote_average": "tmdb_score",
            "vote_count": "tmdb_votes",
        }
    )
    nf["title_key"] = nf["title"].apply(_normalize_title)
    nf["year"] = pd.to_numeric(nf["year"], errors="coerce")

    keep = [
        "title_key",
        "year",
        "content_type",
        "description",
        "genres",
        "cast",
        "director",
        "language",
        "country",
        "tmdb_score",
        "tmdb_votes",
        "popularity",
    ]
    return (
        nf[keep]
        .drop_duplicates(subset=["title_key", "year", "content_type"])
        .reset_index(drop=True)
    )


def _load_disney_content() -> pd.DataFrame:
    """
    Disney titles enriched with aggregated cast / directors from credits.
    """
    titles = pd.read_csv(os.path.join(DATA_DIR, "disney", "titles.csv"))
    credits = pd.read_csv(os.path.join(DATA_DIR, "disney", "credits.csv"))

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
        "title_key",
        "year",
        "content_type",
        "description",
        "genres",
        "cast",
        "director",
        "imdb_score",
        "imdb_votes",
        "tmdb_popularity",
        "tmdb_score",
        "age_certification",
    ]
    return (
        titles[keep]
        .rename(
            columns={
                "imdb_score": "disney_imdb_score",
                "imdb_votes": "disney_imdb_votes",
                "tmdb_popularity": "popularity",
                "age_certification": "disney_age_cert",
            }
        )
        .drop_duplicates(subset=["title_key", "year", "content_type"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_data(include_orphans: bool = True) -> pd.DataFrame:
    """
    Build and return the master recommendation DataFrame.

    Parameters
    ----------
    include_orphans : bool
        When True (default), append Netflix- and Disney-sourced titles that have
        no entry in the platform base datasets (mostly post-2021 content).  These
        rows get platform flags inferred from their source.

    Returns
    -------
    pd.DataFrame – final columns (plus any extras from enrichment sources):
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

    # ----- Enrich base with Netflix rich content -----
    # Netflix content columns (excluding join keys and content_type already in base)
    nf_content = netflix.drop(columns=["content_type"])
    df = _coalesce_merge(base, nf_content, on=["title_key", "year"])

    # ----- Enrich remaining gaps with Disney content -----
    # Separate Disney-specific score cols from content cols so they don't clobber
    ds_content = disney.drop(columns=["content_type"])
    df = _coalesce_merge(df, ds_content, on=["title_key", "year"])

    # Backfill imdb_score from Disney's imdb score where still missing
    if "disney_imdb_score" in df.columns:
        df["imdb_score"] = df["imdb_score"].combine_first(df["disney_imdb_score"])
        df.drop(columns=["disney_imdb_score"], inplace=True)

    # Rename Disney imdb_votes to a unified column
    if "disney_imdb_votes" in df.columns:
        df.rename(columns={"disney_imdb_votes": "imdb_votes"}, inplace=True)

    # Backfill age_rating from Disney age cert
    if "disney_age_cert" in df.columns:
        df["age_rating"] = df["age_rating"].combine_first(df["disney_age_cert"])
        df.drop(columns=["disney_age_cert"], inplace=True)

    # ----- Optionally append orphan titles (post-2021, not in platform base) -----
    if include_orphans:
        base_keys = set(zip(base["title_key"], base["year"]))

        def _make_orphans(
            source: pd.DataFrame, orig_titles: pd.DataFrame, platform_col: str
        ) -> pd.DataFrame:
            """Source rows not already in the platform base."""
            mask = ~source.apply(lambda r: (r["title_key"], r["year"]) in base_keys, axis=1)
            orphans = orig_titles[mask].copy()
            orphans["title"] = orig_titles.loc[mask, "title_key"]
            for col in ("netflix", "hulu", "prime_video", "disney_plus"):
                orphans[col] = 1 if col == platform_col else 0
            return orphans

        # Need original title (not title_key) for orphan rows
        nf_with_title = netflix.copy()
        nf_movies_raw = pd.read_csv(
            os.path.join(DATA_DIR, "netflix", "netflix_movies_detailed_up_to_2025.csv")
        )
        nf_tv_raw = pd.read_csv(
            os.path.join(DATA_DIR, "netflix", "netflix_tv_shows_detailed_up_to_2025.csv")
        )
        nf_titles_lookup = pd.concat([nf_movies_raw[["title"]], nf_tv_raw[["title"]]])
        nf_titles_lookup["title_key"] = nf_titles_lookup["title"].apply(_normalize_title)
        nf_with_title = netflix.merge(
            nf_titles_lookup.drop_duplicates("title_key"), on="title_key", how="left"
        )

        ds_with_title = pd.read_csv(os.path.join(DATA_DIR, "disney", "titles.csv"))[
            ["title"]
        ].copy()
        ds_with_title["title_key"] = ds_with_title["title"].apply(_normalize_title)
        ds_full = disney.merge(
            ds_with_title.drop_duplicates("title_key"), on="title_key", how="left"
        )

        nf_orphans = _make_orphans(nf_with_title, nf_with_title, "netflix")
        ds_orphans = _make_orphans(ds_full, ds_full, "disney_plus")

        # Align orphan columns to df before concat
        for col in df.columns:
            for orph in (nf_orphans, ds_orphans):
                if col not in orph.columns:
                    orph[col] = None
        nf_orphans = nf_orphans[[c for c in df.columns if c in nf_orphans.columns]]
        ds_orphans = ds_orphans[[c for c in df.columns if c in ds_orphans.columns]]

        df = pd.concat([df, nf_orphans, ds_orphans], ignore_index=True)

    # ----- Final clean-up -----

    # Drop join key
    df.drop(columns=["title_key"], inplace=True, errors="ignore")

    # Remove any accidentally duplicated columns from merges
    df = df.loc[:, ~df.columns.duplicated()]

    # Deduplicate keeping the row with the most non-null values
    df["_key"] = df["title"].apply(_normalize_title)
    df["_nulls"] = df.isnull().sum(axis=1)
    df = (
        df.sort_values("_nulls")
        .drop_duplicates(subset=["_key", "year", "content_type"], keep="first")
        .drop(columns=["_key", "_nulls"])
    )

    # Normalise platform flags to 0/1
    for col in ("netflix", "hulu", "prime_video", "disney_plus"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # ----- Build TF-IDF soup -----
    def _build_soup(row) -> str:
        parts = []
        if row.get("description") and pd.notna(row["description"]):
            parts.append(str(row["description"]))
        if row.get("genres") and pd.notna(row["genres"]):
            g = str(row["genres"]).replace(",", " ")
            parts.extend([g, g])  # double-weight genres
        if row.get("cast") and pd.notna(row["cast"]):
            parts.append(str(row["cast"]))
        if row.get("director") and pd.notna(row["director"]):
            parts.append(str(row["director"]))
        return " ".join(parts).strip()

    df["tfidf_soup"] = df.apply(_build_soup, axis=1)

    # ----- Canonical column order -----
    ordered = [
        "title",
        "year",
        "content_type",
        "age_rating",
        "netflix",
        "hulu",
        "prime_video",
        "disney_plus",
        "description",
        "genres",
        "cast",
        "director",
        "language",
        "country",
        "imdb_score",
        "imdb_votes",
        "rotten_tomatoes",
        "tmdb_score",
        "tmdb_votes",
        "popularity",
        "tfidf_soup",
    ]
    present = [c for c in ordered if c in df.columns]
    extra = [c for c in df.columns if c not in ordered]
    return df[present + extra].reset_index(drop=True)


if __name__ == "__main__":
    df = load_data()
    print(f"Final shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print("\nPlatform counts:")
    for col in ("netflix", "hulu", "prime_video", "disney_plus"):
        print(f"  {col}: {df[col].sum()}")
    print(f"\nContent type breakdown:\n{df['content_type'].value_counts()}")
    print("\nNull counts in key TF-IDF columns:")
    key_cols = ["description", "genres", "cast", "director", "tfidf_soup"]
    print(df[key_cols].isnull().sum())
    print(f"\nSample tfidf_soup (row 0):\n{df['tfidf_soup'].iloc[0][:300]}")
    sample_cols = [
        "title",
        "year",
        "content_type",
        "netflix",
        "hulu",
        "prime_video",
        "disney_plus",
        "genres",
        "imdb_score",
    ]
    print(f"\nSample row:\n{df.iloc[0][sample_cols].to_dict()}")
