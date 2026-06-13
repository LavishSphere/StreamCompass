"""
recommender.py — Content-based recommendation engine for StreamCompass.

Each stage of the pipeline maps to a concept from CS 4100:

  Stage 1  TF-IDF vectorisation   — core text representation
  Stage 2  UCS retrieval          — Search: Uniform Cost Search on a similarity graph
  Stage 3  Bayesian scoring       — Uncertainty & Probability: prior x likelihood
  Stage 4  LinearScorer           — Linear Regression + Gradient Descent
  Stage 5  sigmoid / relevance    — Classification: logistic / sigmoid function
  Stage 6  SimpleNN               — Neural Networks: 2-layer feed-forward scorer
  Stage 7  MDP re-ranking         — Markov Decision Process: greedy diversity policy

Usage:
    from data_loader import load_data
    from recommender import build_tfidf_index, recommend

    df = load_data()
    vectorizer, tfidf_matrix = build_tfidf_index(df)
    results = recommend("space adventure sci-fi", df, vectorizer, tfidf_matrix)
"""

import heapq
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Stage 1 — TF-IDF index
# ---------------------------------------------------------------------------


def build_tfidf_index(df: pd.DataFrame):
    """
    Build a TF-IDF matrix over the pre-built `tfidf_soup` column.

    TF-IDF (Term Frequency x Inverse Document Frequency) assigns each word a
    weight that reflects how important it is to a document relative to the
    whole corpus.  Words like "the" get near-zero weight; genre labels that
    were double-weighted in data_loader score higher.

    Returns
    -------
    vectorizer   : fitted TfidfVectorizer  (needed to encode arbitrary queries)
    tfidf_matrix : sparse (n_titles, n_features) matrix
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),  # unigrams + bigrams capture phrases like "sci fi"
        max_features=20_000,
        sublinear_tf=True,  # replace raw TF with 1 + log(TF) to dampen outliers
    )
    tfidf_matrix = vectorizer.fit_transform(df["tfidf_soup"].fillna(""))
    return vectorizer, tfidf_matrix


# ---------------------------------------------------------------------------
# Stage 2 — Uniform Cost Search (UCS) candidate retrieval
# ---------------------------------------------------------------------------


def ucs_retrieve(query_vec, tfidf_matrix, n_candidates: int = 50):
    """
    UNIFORM COST SEARCH — CS 4100 Search Algorithms
    =================================================
    Frame retrieval as a single-source shortest-path problem:

      Nodes : every title in the corpus
      Start : the encoded query vector (not in the corpus)
      Edges : query -> title_i  with cost = 1 - cosine_similarity(query, title_i)
      Goal  : collect the n_candidates nodes with the lowest total cost

    UCS uses a min-priority queue (binary heap) and expands nodes in order of
    non-decreasing cost, guaranteeing we find the most-similar titles first.
    On a star-shaped graph (query to every node, no indirect paths) this
    reduces to a heap-sort by similarity — but the UCS framing generalises
    cleanly to multi-hop "similar-to-similar" graphs if needed.

    Time complexity: O(n log n) for heapify + n pops.

    Returns list of (corpus_index, similarity_score) sorted descending.
    """
    # Cosine similarity: inner product of unit-normalised TF-IDF vectors.
    # Shape: (n_titles,)
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Priority queue: cost = 1 - similarity so the most similar title has
    # the lowest cost (cost 0 = identical, cost 1 = completely unrelated).
    heap = [(1.0 - float(s), i) for i, s in enumerate(sims) if s > 0.0]
    heapq.heapify(heap)  # O(n) in-place

    visited = set()
    candidates = []

    # Expand nodes cheapest-first — identical to the UCS frontier expansion.
    while heap and len(candidates) < n_candidates:
        cost, idx = heapq.heappop(heap)  # O(log n)
        if idx not in visited:
            visited.add(idx)
            candidates.append((idx, 1.0 - cost))  # store as (index, similarity)

    return candidates  # sorted by similarity descending


# ---------------------------------------------------------------------------
# Stage 3 — Bayesian / probabilistic scoring
# ---------------------------------------------------------------------------


def bayesian_score(similarity: float, imdb_score, imdb_weight: float = 0.25) -> float:
    """
    UNCERTAINTY & PROBABILITY — CS 4100
    =====================================
    Combines two independent evidence signals with a linear opinion pool:

        score = (1 - w) * P(content_match) + w * P(quality)

    where:
      P(content_match) = cosine_similarity     -- likelihood the content fits
      P(quality)       = imdb_score / 10.0     -- prior belief title is good
      w                = imdb_weight           -- trust in the prior

    When IMDb score is missing we fall back to the likelihood alone, avoiding
    a hard failure on titles with no rating data.
    """
    if pd.isna(imdb_score) or imdb_score is None or float(imdb_score) <= 0:
        return float(similarity)

    quality_prior = float(imdb_score) / 10.0  # normalise to [0, 1]
    return (1.0 - imdb_weight) * float(similarity) + imdb_weight * quality_prior


# ---------------------------------------------------------------------------
# Stage 4 — Linear scorer with gradient descent weight optimiser
# ---------------------------------------------------------------------------


class LinearScorer:
    """
    LINEAR REGRESSION + GRADIENT DESCENT — CS 4100
    =================================================
    Relevance score as a weighted linear combination of features:

        score(x) = w0*similarity + w1*imdb_norm + w2*popularity_norm
                 = w^T x

    Weights are initialized from domain knowledge (similarity dominates).
    The `gradient_step` method shows how they would be refined via gradient
    descent on MSE loss if user watch/skip labels were available:

        L(w)   = (1/N) sum (w^T x_i - y_i)^2
        dL/dw  = (2/N) X^T (Xw - y)
        w     <- w - alpha * dL/dw
    """

    def __init__(self, weights=None, learning_rate: float = 0.01):
        # [similarity_weight, imdb_weight, popularity_weight]
        self.weights = np.array(weights or [0.60, 0.30, 0.10], dtype=float)
        self.lr = learning_rate

    def score(self, features: np.ndarray) -> np.ndarray:
        """Forward pass: score = X * w  (shape: n_samples,)."""
        return features @ self.weights

    def gradient_step(self, features: np.ndarray, targets: np.ndarray):
        """
        One gradient-descent step on MSE.
        Call in a loop on batches of (features, user_ratings) to personalise.
        """
        predictions = self.score(features)
        residuals = predictions - targets
        grad = (2.0 / len(features)) * features.T @ residuals
        self.weights -= self.lr * grad
        return self  # allow method chaining


# ---------------------------------------------------------------------------
# Stage 5 — Sigmoid / logistic relevance probability
# ---------------------------------------------------------------------------


def sigmoid(x: float) -> float:
    """
    LOGISTIC REGRESSION / CLASSIFICATION — CS 4100
    =================================================
    sigma(x) = 1 / (1 + e^(-x))

    The sigmoid is the activation function of logistic regression.  It
    squashes any real-valued score into (0, 1), making it interpretable as a
    probability of relevance.  Decision boundary sits at x = 0 (sigma = 0.5).
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def logistic_relevance(scores: np.ndarray) -> np.ndarray:
    """
    Convert raw scores to relevance probabilities via sigmoid.
    Mean-centres and scales before applying sigmoid so the output is spread
    across (0, 1) rather than saturated near 0 or 1.
    """
    centred = scores - scores.mean()
    std = centred.std() + 1e-9
    scaled = centred / std * 1.5  # stretch to roughly [-3, +3] pre-sigmoid
    return np.array([sigmoid(float(v)) for v in scaled])


# ---------------------------------------------------------------------------
# Stage 6 — Simple neural network scorer
# ---------------------------------------------------------------------------


class SimpleNN:
    """
    NEURAL NETWORKS — CS 4100
    ===========================
    Two-layer feed-forward network for relevance scoring:

        Input  layer : 3 features  [similarity, imdb_norm, popularity_norm]
        Hidden layer : 8 units,    ReLU activation
        Output layer : 1 unit,     Sigmoid activation -> relevance in (0, 1)

    Forward pass:
        h  = ReLU(W1 * x + b1)
        y^ = sigma(W2 * h + b2)

    Weights are initialized to approximate the domain-knowledge linear scorer
    so the network produces sensible results without any training.  In
    production, `fit()` runs mini-batch gradient descent on user interaction
    labels (watched = 1, skipped = 0) to learn non-linear feature interactions
    that the linear scorer cannot express.
    """

    def __init__(self, n_hidden: int = 8, seed: int = 42):
        rng = np.random.default_rng(seed)
        n_in = 3

        # He initialization scales variance for stable ReLU gradients
        self.W1 = rng.standard_normal((n_hidden, n_in)) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(n_hidden)

        # Output layer: uniform positive weights across all hidden units so the
        # network reliably scores higher-feature inputs higher before training.
        # Concentrating weight on a subset of units risks cancellation to 0.5.
        self.W2 = np.ones((1, n_hidden)) / n_hidden
        self.b2 = np.zeros(1)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation: max(0, x) — introduces non-linearity."""
        return np.maximum(0.0, x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for a batch of feature vectors (shape: n_samples x 3).
        Returns relevance probabilities in (0, 1) via sigmoid output.
        """
        h = self._relu(x @ self.W1.T + self.b1)  # (n_samples, n_hidden)
        raw = (h @ self.W2.T + self.b2).flatten()  # (n_samples,)
        return np.array([sigmoid(float(v)) for v in raw])

    def fit(self, features: np.ndarray, labels: np.ndarray, epochs: int = 50, lr: float = 0.05):
        """
        Mini-batch gradient descent on binary cross-entropy loss.
        Pass (feature_matrix, watch_labels) to personalise the model.
        """
        for _ in range(epochs):
            # --- forward ---
            h = self._relu(features @ self.W1.T + self.b1)
            y_hat = np.array([sigmoid(float(v)) for v in (h @ self.W2.T + self.b2).flatten()])
            # --- output layer gradient ---
            d_out = (y_hat - labels) / len(labels)
            self.W2 -= lr * (d_out[:, None] * h).mean(axis=0, keepdims=True)
            self.b2 -= lr * d_out.mean()
            # --- hidden layer gradient (ReLU gate) ---
            d_h = d_out[:, None] @ self.W2 * (h > 0).astype(float)
            self.W1 -= lr * (d_h.T @ features) / len(features)
            self.b1 -= lr * d_h.mean(axis=0)
        return self


# ---------------------------------------------------------------------------
# Stage 7 — MDP diversity re-ranking
# ---------------------------------------------------------------------------


def mdp_rerank(candidates: list, tfidf_matrix, top_k: int = 10, lambda_div: float = 0.3) -> list:
    """
    MARKOV DECISION PROCESS — CS 4100
    ====================================
    Models the sequential selection of recommendations as a finite-horizon MDP:

      State  s_t : set of t titles already recommended
      Action a   : choose a title from the remaining candidates
      Reward R   : R(s, a) = relevance(a) - lambda * max_{j in s} cos_sim(a, j)
                  (reward = query relevance  MINUS  redundancy with chosen titles)
      Policy pi  : greedy — pick the action maximising immediate reward

    The greedy policy is approximately optimal here because rewards are
    decomposable — the diversity penalty only subtracts, never defers value.
    lambda_div controls the exploration/exploitation trade-off:
        lambda = 0  -> pure relevance (no diversity)
        lambda = 1  -> maximum diversity
    """
    if not candidates:
        return []

    selected = []  # (corpus_index, final_score)
    selected_vecs = []  # sparse row vectors of already-chosen titles
    remaining = list(candidates)

    for _ in range(min(top_k, len(remaining))):
        best_reward = -np.inf
        best_pos = -1

        for pos, (idx, relevance) in enumerate(remaining):
            if selected_vecs:
                # Diversity penalty: max similarity to any already-chosen title
                stacked = np.vstack([v.toarray() for v in selected_vecs])
                div_penalty = float(cosine_similarity(tfidf_matrix[idx], stacked).max())
            else:
                div_penalty = 0.0  # first pick has no redundancy cost

            # MDP immediate reward: R(s, a)
            reward = relevance - lambda_div * div_penalty

            if reward > best_reward:
                best_reward = reward
                best_pos = pos

        if best_pos < 0:
            break

        chosen_idx, chosen_score = remaining.pop(best_pos)
        selected.append((chosen_idx, chosen_score))
        selected_vecs.append(tfidf_matrix[chosen_idx])  # keep sparse

    return selected


# ---------------------------------------------------------------------------
# Feature builder (shared by linear scorer and neural network)
# ---------------------------------------------------------------------------


def _build_features(candidates: list, df: pd.DataFrame) -> np.ndarray:
    """
    Build an (n_candidates x 3) feature matrix:
        col 0 : cosine similarity       (already in [0, 1])
        col 1 : IMDb score / 10         (normalised to [0, 1])
        col 2 : TMDB popularity score   (min-max normalised)

    Missing values are replaced with the column median before normalisation.
    """
    rows = []
    for idx, sim in candidates:
        row = df.iloc[idx]
        imdb = row.get("imdb_score")
        pop = row.get("popularity")
        rows.append(
            [
                float(sim),
                float(imdb) if pd.notna(imdb) else np.nan,
                float(pop) if pd.notna(pop) else np.nan,
            ]
        )

    features = np.array(rows, dtype=float)

    # Impute missing values with column median (guard against all-NaN columns)
    for col in (1, 2):
        col_vals = features[:, col]
        nan_mask = np.isnan(col_vals)
        if nan_mask.all():
            features[:, col] = 0.0
        elif nan_mask.any():
            features[nan_mask, col] = np.nanmedian(col_vals)

    # Normalise IMDb to [0, 1]
    features[:, 1] = np.clip(features[:, 1] / 10.0, 0.0, 1.0)

    # Min-max normalise popularity
    p_min, p_max = features[:, 2].min(), features[:, 2].max()
    if p_max > p_min:
        features[:, 2] = (features[:, 2] - p_min) / (p_max - p_min)
    else:
        features[:, 2] = 0.0

    return features


# ---------------------------------------------------------------------------
# Title match lookup
# ---------------------------------------------------------------------------


def _find_title_match(query: str, df: pd.DataFrame):
    """
    Check whether the query is itself a title in the corpus.

    Returns the DataFrame index of the best match, or None.
    Precedence:
      1. Exact normalised match  (e.g. "Breaking Bad" -> "breaking bad")
      2. First substring match   (e.g. "breaking bad s" still hits it)
    """

    def _norm(t):
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(t).lower())).strip()

    q = _norm(query)
    normalised = df["title"].apply(_norm)

    # Exact match first
    exact = normalised[normalised == q]
    if not exact.empty:
        return exact.index[0]

    # Substring match: query fully contained in a title
    sub = normalised[normalised.str.contains(q, regex=False)]
    if not sub.empty:
        # Prefer the shortest title (least extra words = closest match)
        return sub.apply(len).idxmin()  # returns integer DataFrame index

    return None


# ---------------------------------------------------------------------------
# Main recommendation pipeline
# ---------------------------------------------------------------------------

_RESULT_COLS = [
    "title",
    "year",
    "content_type",
    "genres",
    "imdb_score",
    "netflix",
    "hulu",
    "prime_video",
    "disney_plus",
    "similarity_score",
]


def recommend(
    query: str,
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    top_k: int = 10,
    content_type: str = None,
    platforms: list = None,
    min_imdb: float = None,
    lambda_div: float = 0.3,
) -> pd.DataFrame:
    """
    End-to-end recommendation pipeline.

    Pipeline stages
    ---------------
    1. Encode query   -> TF-IDF vector                     (TF-IDF)
    2. UCS retrieval  -> top candidates from corpus        (Search / UCS)
    3. Hard filters   -> content_type / platform / IMDb
    4. Bayesian score -> blend similarity + IMDb prior     (Probability)
    5. Linear scorer  -> weighted feature combination      (Linear Regression)
    6. Sigmoid        -> relevance probabilities           (Logistic Regression)
    7. Neural network -> non-linear re-scoring             (Neural Networks)
    8. MDP re-rank    -> diversity-aware final selection   (MDP)

    Parameters
    ----------
    query        : free-text search string
    df           : master DataFrame from data_loader.load_data()
    vectorizer   : fitted TfidfVectorizer from build_tfidf_index()
    tfidf_matrix : sparse TF-IDF matrix from build_tfidf_index()
    top_k        : number of results to return (default 10)
    content_type : "movie" or "tv" (optional filter)
    platforms    : list of platform column names to filter on,
                   e.g. ["netflix", "disney_plus"]
    min_imdb     : minimum IMDb score threshold (optional)
    lambda_div   : MDP diversity weight in [0, 1]

    Returns
    -------
    pd.DataFrame with columns defined in _RESULT_COLS
    """
    # Guard: empty query
    if not query or not query.strip():
        return pd.DataFrame(columns=_RESULT_COLS)

    # ---- Title match: if query is a known title, seed from its TF-IDF row ----
    # This ensures "Breaking Bad" uses Breaking Bad's own description/cast/genres
    # as the similarity seed rather than the raw words "breaking" and "bad".
    # Falls back to text encoding if the matched title has no content (empty soup).
    title_match_idx = _find_title_match(query.strip(), df)
    if title_match_idx is not None and tfidf_matrix[title_match_idx].nnz > 0:
        query_vec = tfidf_matrix[title_match_idx]
    else:
        title_match_idx = None  # treat as plain text query
        # ---- Stage 1: encode free-text query into TF-IDF space ----
        query_vec = vectorizer.transform([query.strip()])
        if query_vec.nnz == 0:
            # All query terms are unknown — zero vector, cannot retrieve anything
            return pd.DataFrame(columns=_RESULT_COLS)

    # ---- Stage 2: UCS — retrieve candidates from similarity graph ----
    # Over-fetch (5x top_k) to give downstream stages room to filter/re-rank
    candidates = ucs_retrieve(query_vec, tfidf_matrix, n_candidates=top_k * 5)

    # Exclude the matched title itself from results
    if title_match_idx is not None:
        candidates = [(i, s) for i, s in candidates if i != title_match_idx]
    if not candidates:
        return pd.DataFrame(columns=_RESULT_COLS)

    # ---- Stage 3: hard filters ----
    if content_type:
        candidates = [
            (i, s)
            for i, s in candidates
            if df.iloc[i].get("content_type", "") == content_type.lower()
        ]
    if platforms:
        candidates = [
            (i, s)
            for i, s in candidates
            if any(int(df.iloc[i].get(p) or 0) == 1 for p in platforms)
        ]
    if min_imdb is not None:
        candidates = [
            (i, s)
            for i, s in candidates
            if pd.notna(df.iloc[i].get("imdb_score"))
            and float(df.iloc[i]["imdb_score"]) >= min_imdb
        ]
    if not candidates:
        return pd.DataFrame(columns=_RESULT_COLS)

    # ---- Stage 4: Bayesian scoring ----
    candidates = [(i, bayesian_score(s, df.iloc[i].get("imdb_score"))) for i, s in candidates]

    # ---- Stage 5: Linear scorer ----
    features = _build_features(candidates, df)
    linear_scores = LinearScorer().score(features)
    # Blend Bayesian and linear scores with equal weight
    candidates = [
        (idx, 0.5 * bay + 0.5 * float(lin)) for (idx, bay), lin in zip(candidates, linear_scores)
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    candidates = candidates[: top_k * 3]  # trim before neural pass

    # ---- Stage 6: sigmoid — convert to relevance probabilities ----
    raw_scores = np.array([s for _, s in candidates])
    relevance_probs = logistic_relevance(raw_scores)
    candidates = [(idx, float(p)) for (idx, _), p in zip(candidates, relevance_probs)]

    # ---- Stage 7: neural network — non-linear re-scoring ----
    nn_features = _build_features(candidates, df)
    nn_scores = SimpleNN().forward(nn_features)
    # Blend NN output with logistic relevance probability
    candidates = [
        (idx, 0.5 * float(prob) + 0.5 * float(nn_s))
        for (idx, prob), nn_s in zip(candidates, nn_scores)
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)

    # ---- Stage 8: MDP diversity re-ranking ----
    final = mdp_rerank(candidates, tfidf_matrix, top_k=top_k, lambda_div=lambda_div)

    return _format_results(final, df)


def _format_results(ranked: list, df: pd.DataFrame) -> pd.DataFrame:
    """Pack ranked (index, score) pairs into a clean result DataFrame."""
    rows = []
    for idx, score in ranked:
        row = df.iloc[idx]
        rows.append(
            {
                "title": row.get("title"),
                "year": row.get("year"),
                "content_type": row.get("content_type"),
                "genres": row.get("genres"),
                "imdb_score": row.get("imdb_score"),
                "netflix": int(row.get("netflix") or 0),
                "hulu": int(row.get("hulu") or 0),
                "prime_video": int(row.get("prime_video") or 0),
                "disney_plus": int(row.get("disney_plus") or 0),
                "similarity_score": round(score, 4),
            }
        )
    return pd.DataFrame(rows, columns=_RESULT_COLS)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_loader import load_data

    print("Loading data...")
    _df = load_data()
    print(f"Loaded {len(_df):,} titles.")

    print("Building TF-IDF index...")
    _vec, _mat = build_tfidf_index(_df)
    print(f"Index: {_mat.shape[0]:,} docs x {_mat.shape[1]:,} features\n")

    _queries = [
        "dark psychological thriller mystery",
        "animated family adventure",
        "Breaking Bad",
        "romantic comedy New York",
    ]

    for _q in _queries:
        print(f"=== '{_q}' ===")
        _res = recommend(_q, _df, _vec, _mat)
        print(_res.to_string(index=False))
        print()
