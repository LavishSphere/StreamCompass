# 🧭 StreamCompass

> Navigate the streaming landscape - find your next watch and where to watch it.

StreamCompass is a content-based recommendation engine that takes a show or movie you love and surfaces similar titles across streaming platforms. Tell us what you just watched, and we'll point you to what's next.

---

## Features

- **Title Search** - search any movie or show to use as your starting point
- **Smart Recommendations** - similarity scoring based on genre, cast, tags, and more
- **Platform Mapping** - see which streaming services carry each recommended title
- **Filter & Explore** - filter results by platform, content type, or minimum IMDb score
- **Match Breakdown** - hover any similarity score to see exactly why a title was recommended (description, genre, cast, director contributions)
- **Clean UI** - dark-mode interface built for the streaming experience

---

## Tech Stack

**Frontend**
- React + Vite
- React Router

**Backend**
- FastAPI (Python)
- scikit-learn (TF-IDF + cosine similarity)
- pandas / numpy
- scipy

**Data**
- [MovieLens 32M Dataset](https://grouplens.org/datasets/movielens/) for movie genres, user tags, and cached aggregate ratings
- Streaming platform availability data via Kaggle
- Netflix and Disney+ detailed metadata (genres, cast, directors, descriptions)

---

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

The API will be running at `http://localhost:8000`. Docs available at `http://localhost:8000/docs`.

### MovieLens Data

StreamCompass includes MovieLens 32M metadata under `backend/data/ml-32m/` for movie genres, user tags, and cached aggregate rating signals. The app automatically uses `movies.csv`, `tags.csv`, `links.csv`, and `ratings_summary.csv` during backend startup.

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The app will be running at `http://localhost:5173`.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/search?q={title}` | Search for a title |
| POST | `/recommend` | Get recommendations for a given title |
| GET | `/platforms` | List all supported streaming platforms |
| GET | `/genres` | List available genres |
| GET | `/title/{title}` | Get metadata for a title |
| GET | `/health` | Check API/data load status |

---

## Problem Statement

Given a title a user has already watched and enjoyed, StreamCompass identifies the most similar titles available on streaming platforms and ranks them by content similarity. The system handles  ~43,000 titles across Netflix, Hulu, Prime Video, and Disney+, and return results in real time via a REST API.

This is a **content-based filtering** problem: we have no user interaction history or ratings, so recommendations are derived from four content signals: plot description and user tags (50%), genres (30%), cast (10%), and director (10%). The challenge is building a representation of each title rich enough to capture meaningful similarity, and a ranking pipeline sophisticated enough to surface relevant results while also avoiding redundancy.

---

## AI Techniques Used

The recommendation pipeline in `recommender.py` implements several distinct AI/ML concepts from CS 4100, applied sequentially as a staged pipeline:

### Stage 1 — TF-IDF Vectorization (Text Representation)

Each title is represented as a weighted vector across a vocabulary of ~48,000 terms (unigrams and bigrams). We used separate TF-IDF matrices for description, genres, cast, and director, and combined them with explicit weights:

```
description: 0.50  |  genres: 0.30  |  cast: 0.10  |  director: 0.10
```

TF-IDF (Term Frequency × Inverse Document Frequency) assigns high weight to terms that are distinctive to a specific title and low weight to terms that appear across many titles. The result is a sparse high-dimensional vector that captures the content of each title. Sublinear TF scaling (`1 + log(TF)`) also dampens the effect of repeated terms. This is how we chose to represent our title text from just letters into a vector.

### Stage 2 — Uniform Cost Search (UCS) Candidate Retrieval

Retrieval is framed as a search problem. We model the data/corpus as a star-shaped graph where every title is a node connected to the query with an edge cost of `1 − cosine_similarity`. UCS expands nodes in order of non-decreasing cost using a min-priority heap, guaranteeing that the most similar titles are found first.

- **State space**: all ~43,000 titles in the corpus
- **Start node**: encoded query vector
- **Edge cost**: `1 − cosine_similarity(query, title)`
- **Goal**: collect the top-N lowest-cost nodes
- **Time complexity**: O(n log n)

This framing generalizes naturally to multi-hop "similar-to-similar" graph structures if needed.

### Stage 3 — Bayesian Scoring (Uncertainty & Probability)

After retrieval, we combine two independent evidence signals using a linear opinion pool:

```
score = (1 − w) × P(content_match) + w × P(quality)
```

where `P(content_match)` is the cosine similarity (likelihood the content fits) and `P(quality)` is the normalized IMDb score (prior belief the title is good). When IMDb data is missing, the system falls back to similarity alone. This helps prevent hard failures on titles with no rating data.

### Stage 4 — Linear Scorer with Gradient Descent

A weighted linear combination of three features produces a relevance score:

```
score(x) = 0.75 × tfidf_similarity + 0.20 × imdb_norm + 0.05 × popularity_norm
```

The `LinearScorer` class also implements a `gradient_step` method that shows how weights would be updated via gradient descent on MSE loss if user watch/skip labels were available:

```
L(w) = (1/N) Σ (wᵀxᵢ − yᵢ)²
dL/dw = (2/N) Xᵀ(Xw − y)
w ← w − α × dL/dw
```

### Stage 5 — Logistic Regression / Sigmoid

Raw scores are converted to relevance probabilities using the sigmoid function:

```
σ(x) = 1 / (1 + e^(−x))
```

The sigmoid is the activation function of logistic regression. Scores are mean-centered and scaled before applying sigmoid so the output is spread across (0, 1) rather than saturated near the extremes. We chose the sigmoid because it makes scores interpretable as a **probability** of relevance.

### Stage 6 — Neural Network Re-Scoring

A two-layer feed-forward neural network re-scores candidates using non-linear feature interactions:

```
h  = ReLU(W₁x + b₁)     # hidden layer: 8 units
ŷ  = σ(W₂h + b₂)        # output: relevance probability in (0, 1)
```

Weights are initialized with He initialization for stable ReLU gradients. The network has a `fit()` method for mini-batch gradient descent on binary cross-entropy loss if user interaction data becomes available. Currently it runs with domain-knowledge initialization but already captures non-linear interactions that the linear scorer cannot express.

### Stage 7 — MDP Diversity Re-Ranking (Markov Decision Process)

The final selection is modeled as a MDP:

- **State** `sₜ`: set of t titles already selected
- **Action**: choose a title from remaining candidates
- **Reward**: `R(s, a) = relevance(a) − λ × max_{j∈s} cos_sim(a, j)`

The greedy policy picks the action maximizing immediate reward at each step. The diversity weight `λ` (default 0.1) controls the relevance/diversity tradeoff — at 0 the system returns the most similar titles; at 1 it maximizes variety. This prevents the top-10 results from being near-identical titles by the same director.

---

## System Design

### Data Pipeline (`data_loader.py`)

The dataset is assembled from five sources at server startup:

| Source | Contents |
|--------|----------|
| `MoviesOnStreamingPlatforms.csv` | Platform availability flags for movies |
| `tv_shows.csv` | Platform availability flags for TV shows |
| `netflix/` (movies + TV) | Rich metadata: descriptions, genres, cast, directors, TMDB scores |
| `disney/titles.csv + credits.csv` | Disney+ titles with cast/director credits joined |
| `ml-32m/` (MovieLens 32M) | 87k movies with pipe-delimited genres, user tags, and aggregate ratings |

The merge strategy is **coalesce merging**: sources are joined left-to-right on `(title_key, year)`, and each new source fills in gaps from the previous one without overwriting existing data. This maximizes coverage while preventing higher-quality sources from being clobbered by lower-quality ones.

MovieLens tags are appended to descriptions and MovieLens genres are merged into the genre field, giving the TF-IDF model richer signals for titles that previously had sparse metadata. The Amazon Prime dataset was excluded because it was found to contain fabricated titles and directors.

A `tfidf_soup` column is built for each title by concatenating all metadata fields, which serves as the input to the vectorizer.

### API Layer (`main.py`)

The FastAPI application loads the dataset and builds the TF-IDF index once at startup using a `lifespan` context manager, storing everything in a shared `_state` dictionary. This ensures the expensive vectorization step (fitting on ~43k documents) only happens once rather than on every request.

The `/recommend` endpoint accepts:

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | Title or free-text query |
| `top_k` | int (1–50) | Number of results to return |
| `content_type` | string | Filter to `movie` or `tv` |
| `platforms` | list | Filter to specific platforms |
| `min_imdb` | float | Minimum IMDb score threshold |
| `lambda_div` | float | MDP diversity weight (0–1) |

The response includes a `match_breakdown` field for each result containing per-field TF-IDF scores and quality signals, which powers the frontend's hover tooltip.

---

## Parameter Choices and Tuning

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| TF-IDF field weights | desc: 0.50, genres: 0.30, cast: 0.10, dir: 0.10 | Description carries the most semantic signal; genres are a strong structural signal; cast/director are useful for style but noisier |
| `max_features` | 12,000 per field | Balances vocabulary coverage against memory; 4 matrices × 12k features = ~48k total features |
| `ngram_range` | (1, 2) | Bigrams capture phrases like "sci fi" and "based on" that unigrams miss |
| `sublinear_tf` | True | Dampens the effect of frequently repeated words within a title |
| Linear scorer weights | 0.75 / 0.20 / 0.05 | Content similarity dominates; IMDb and popularity act as tiebreakers |
| `lambda_div` default | 0.1 | Gentle diversity — avoids near-duplicates while keeping results close to the query |
| `top_k` default | 10 (API), 40 (frontend) | Frontend over-fetches to populate multiple row groupings |
| `n_candidates` (UCS) | `top_k × 5` | Over-fetch at retrieval stage to give downstream stages room to filter and re-rank |

---

## Testing and Evaluation

Qualitative testing was performed by querying known titles and manually evaluating result coherence:

- `"Inception"` → returns Interstellar, The Matrix, Arrival, Severance, Blade Runner 2049 — all thematically aligned (mind-bending, sci-fi, cerebral)
- `"Breaking Bad"` → returns Better Call Saul, Ozark, Narcos, The Wire — all crime dramas with morally complex protagonists
- `"dark psychological thriller mystery"` (free-text) → returns titles across genres that match the descriptive terms even without a specific title match

**API testing** via `curl` confirmed correct JSON responses for all six endpoints, accurate platform flag encoding (0/1 integers), and graceful null handling for titles with missing IMDb scores.

**Edge cases tested:**
- Queries with no matching title (falls back to free-text TF-IDF encoding)
- Titles with empty `tfidf_soup` (falls back to title string)
- Platform filters that produce zero results (returns empty results array, no 500 error)
- NaN values in IMDb/popularity fields (imputed with column median before scoring)

---

## Limitations and Lessons Learned

**Dataset quality**: A significant portion of titles in the dataset have missing or sparse metadata. Titles without descriptions rely entirely on genre and cast for similarity, which produces coarser recommendations. The Amazon Prime dataset was discovered to contain entirely fabricated data and had to be excluded.

**Cold-start behavior**: Content-based filtering has no cold-start problem for new titles (we can recommend as soon as metadata exists), but it cannot learn from user preferences without interaction data. Two users with opposite tastes will receive identical recommendations for the same query.

**Popularity bias**: Titles with richer metadata (more cast listed, longer descriptions, more user tags from MovieLens) tend to rank higher because they produce denser TF-IDF vectors. Lesser-known titles with sparse metadata are systematically disadvantaged.

**TF-IDF limitations**: The model has no understanding of semantic meaning — it matches on word overlap, not concepts. "Space exploration" and "astronaut adventure" are not recognized as similar unless the exact words co-occur. A dense embedding model (e.g. sentence-transformers) would handle this better but at higher computational cost.

**Diversity tuning**: The MDP diversity re-ranker was found to hurt relevance more than it helped variety at higher `lambda_div` values. The default was reduced from 0.3 to 0.1, and the re-ranker is architecturally present but effectively minimal in impact. A learned diversity weight trained on user feedback would be more principled.

---

## Connection to CS4100 Concepts

| CS4100 Topic | Application in StreamCompass |
|---------------|------------------------------|
| Search (BFS/DFS/UCS/A*) | UCS retrieval over a similarity graph to find top-N candidates |
| Uncertainty & Probability | Bayesian scoring combining content likelihood with IMDb quality prior |
| Markov Decision Processes | Greedy MDP policy for diversity-aware final re-ranking |
| Linear Regression | LinearScorer with gradient descent weight optimization |
| Logistic Regression | Sigmoid activation converting raw scores to relevance probabilities |
| Neural Networks | Two-layer feed-forward network with ReLU hidden layer and sigmoid output |
| Text Representation | TF-IDF vectorization with per-field weighting and bigram support |

---

## Project Structure

```
StreamCompass/
├── backend/
│   ├── main.py            # FastAPI app and routes
│   ├── recommender.py     # 7-stage recommendation pipeline
│   ├── data_loader.py     # Dataset loading, merging, and preprocessing
│   ├── requirements.txt
│   └── data/              # Dataset files (CSV sources + MovieLens)
├── frontend/
│   ├── src/
│   │   ├── App.jsx        # Browse/results page with filters and drawer
│   │   ├── LandingPage.jsx # Search entry point with autocomplete
│   │   └── main.jsx       # React entry point and routing
│   ├── index.html
│   └── package.json
└── README.md
```

---

## Screenshots

**Figure 1 — Search results for "Inception"**
ADD IMAGE HERE ONCE DONE

**Figure 2 — Match breakdown tooltip showing TF-IDF field contributions**
ADD IMAGE HERE ONCE DONE

---

## Team

| Name | Role |
|------|------|
| Ashsmith Khayrul | ML & Backend |
| Angie Che | Frontend & Integration |

---

## Course

CS 4100 - Artificial Intelligence
Northeastern University, Summer 2026