"""
main.py — FastAPI application for StreamCompass.

Loads the dataset and TF-IDF index once on startup, then serves six endpoints:

  GET  /health              — liveness check
  GET  /search              — title autocomplete
  POST /recommend           — core recommendation engine
  GET  /title/{title}       — detail for a single title
  GET  /platforms           — list supported streaming platforms
  GET  /genres              — list available genres for filter UI
"""

from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data_loader import load_data
from recommender import build_tfidf_index, recommend

# ---------------------------------------------------------------------------
# App state — loaded once at startup, shared across all requests
# ---------------------------------------------------------------------------

_state: dict = {}

PLATFORMS = ["netflix", "hulu", "prime_video", "disney_plus"]


def _nn(value):
    """Return None if value is pandas/numpy NaN, otherwise return value as-is."""
    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return value


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load dataset and build TF-IDF index once when the server starts."""
    df = load_data()
    vectorizer, tfidf_matrix = build_tfidf_index(df)
    _state["df"] = df
    _state["vectorizer"] = vectorizer
    _state["tfidf_matrix"] = tfidf_matrix

    # Pre-compute genre list for the /genres endpoint
    all_genres: set = set()
    for raw in df["genres"].dropna():
        for g in str(raw).split(","):
            g = g.strip()
            if g:
                all_genres.add(g)
    _state["genres"] = sorted(all_genres)

    yield

    _state.clear()


app = FastAPI(
    title="StreamCompass API",
    description="Content-based streaming recommendation engine.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://streamcompass.khayrul.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Title or free-text search query")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")
    content_type: Optional[str] = Field(None, description="Filter by content type: 'movie' or 'tv'")
    platforms: Optional[list[str]] = Field(
        None,
        description="Filter to titles on these platforms: netflix, hulu, prime_video, disney_plus",
    )
    min_imdb: Optional[float] = Field(None, ge=0.0, le=10.0, description="Minimum IMDb score")
    lambda_div: float = Field(
        0.1, ge=0.0, le=1.0, description="Diversity weight (0=pure relevance, 1=max diversity)"
    )


class TitleResult(BaseModel):
    title: str
    year: Optional[float]
    content_type: Optional[str]
    genres: Optional[str]
    imdb_score: Optional[float]
    netflix: int
    hulu: int
    prime_video: int
    disney_plus: int
    similarity_score: float
    match_breakdown: Optional[dict] = None


class RecommendResponse(BaseModel):
    query: str
    results: list[TitleResult]
    total: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Meta"])
def health():
    """Liveness check — confirms the API is up and data is loaded."""
    df = _state.get("df")
    return {
        "status": "ok",
        "titles_loaded": len(df) if df is not None else 0,
    }


@app.get("/search", tags=["Search"])
def search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Max results to return"),
):
    """
    Autocomplete / title search.

    Returns titles whose normalised name contains the query string,
    sorted by IMDb score descending. Useful for populating a search
    dropdown before the user fires a full /recommend request.
    """
    df: pd.DataFrame = _state["df"]
    q_norm = q.strip().lower()

    mask = df["title"].str.lower().str.contains(q_norm, na=False, regex=False)
    hits = df[mask].copy()

    # Sort: exact-start matches first, then by IMDb score descending
    hits["_exact"] = hits["title"].str.lower().str.startswith(q_norm).astype(int)
    hits = hits.sort_values(["_exact", "imdb_score"], ascending=[False, False]).head(limit)

    return {
        "query": q,
        "results": [
            {
                "title": row["title"],
                "year": _nn(row.get("year")),
                "content_type": _nn(row.get("content_type")),
                "genres": _nn(row.get("genres")),
                "imdb_score": _nn(row.get("imdb_score")),
                "netflix": int(row.get("netflix") or 0),
                "hulu": int(row.get("hulu") or 0),
                "prime_video": int(row.get("prime_video") or 0),
                "disney_plus": int(row.get("disney_plus") or 0),
            }
            for _, row in hits.iterrows()
        ],
        "total": int(mask.sum()),
    }


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommend"])
def get_recommendations(body: RecommendRequest):
    """
    Core recommendation endpoint.

    Accepts a title or free-text query and returns the top-K most similar
    titles, optionally filtered by content type, platform, and IMDb score.
    lambda_div is accepted for compatibility, but MDP diversity re-ranking is
    currently disabled in the recommender.
    """
    # Validate platform names if provided
    if body.platforms:
        bad = [p for p in body.platforms if p not in PLATFORMS]
        if bad:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown platform(s): {bad}. Valid: {PLATFORMS}",
            )

    if body.content_type and body.content_type.lower() not in ("movie", "tv"):
        raise HTTPException(
            status_code=422,
            detail="content_type must be 'movie' or 'tv'",
        )

    df = _state["df"]
    vectorizer = _state["vectorizer"]
    tfidf_matrix = _state["tfidf_matrix"]

    results_df = recommend(
        query=body.query,
        df=df,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        top_k=body.top_k,
        content_type=body.content_type,
        platforms=body.platforms,
        min_imdb=body.min_imdb,
        lambda_div=body.lambda_div,
    )

    if results_df.empty:
        return RecommendResponse(query=body.query, results=[], total=0)

    results = [
        TitleResult(
            title=str(row["title"]),
            year=_nn(row.get("year")),
            content_type=_nn(row.get("content_type")),
            genres=_nn(row.get("genres")),
            imdb_score=_nn(row.get("imdb_score")),
            netflix=int(row["netflix"]),
            hulu=int(row["hulu"]),
            prime_video=int(row["prime_video"]),
            disney_plus=int(row["disney_plus"]),
            similarity_score=float(row["similarity_score"]),
            match_breakdown=_nn(row.get("match_breakdown")),
        )
        for _, row in results_df.iterrows()
    ]

    return RecommendResponse(query=body.query, results=results, total=len(results))


@app.get("/title/{title}", tags=["Search"])
def get_title(title: str):
    """
    Detail view for a single title.

    Looks up the closest matching title in the dataset and returns its
    full metadata. Used when a user clicks on a result card.
    """
    df: pd.DataFrame = _state["df"]
    t_norm = title.strip().lower()

    # Exact match first, then substring
    mask_exact = df["title"].str.lower() == t_norm
    if mask_exact.any():
        row = df[mask_exact].iloc[0]
    else:
        mask_sub = df["title"].str.lower().str.contains(t_norm, na=False, regex=False)
        if not mask_sub.any():
            raise HTTPException(status_code=404, detail=f"Title '{title}' not found")
        row = df[mask_sub].iloc[0]

    return {
        "title": row["title"],
        "year": _nn(row.get("year")),
        "content_type": _nn(row.get("content_type")),
        "genres": _nn(row.get("genres")),
        "description": _nn(row.get("description")),
        "cast": _nn(row.get("cast")),
        "director": _nn(row.get("director")),
        "language": _nn(row.get("language")),
        "imdb_score": _nn(row.get("imdb_score")),
        "rotten_tomatoes": _nn(row.get("rotten_tomatoes")),
        "age_rating": _nn(row.get("age_rating")),
        "netflix": int(row.get("netflix") or 0),
        "hulu": int(row.get("hulu") or 0),
        "prime_video": int(row.get("prime_video") or 0),
        "disney_plus": int(row.get("disney_plus") or 0),
    }


@app.get("/platforms", tags=["Meta"])
def get_platforms():
    """List all supported streaming platforms."""
    df: pd.DataFrame = _state["df"]
    return {"platforms": [{"id": p, "title_count": int(df[p].sum())} for p in PLATFORMS]}


@app.get("/genres", tags=["Meta"])
def get_genres():
    """List all genres present in the dataset, for populating filter dropdowns."""
    return {"genres": _state["genres"]}
