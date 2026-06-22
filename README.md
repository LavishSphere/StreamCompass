# 🧭 StreamCompass

> Navigate the streaming landscape - find your next watch and where to watch it.

StreamCompass is a content-based recommendation engine that takes a show or movie you love and surfaces similar titles across streaming platforms. Tell us what you just watched, and we'll point you to what's next.

---

## Features

- **Title Search** - search any movie or show to use as your starting point
- **Smart Recommendations** - similarity scoring based on genre, cast, tags, and more
- **Platform Mapping** - see which streaming services carry each recommended title
- **Filter & Explore** - filter results by platform or genre
- **Clean UI** - dark-mode interface built for the streaming experience

---

## Tech Stack

**Frontend**
- React + Vite
- Tailwind CSS

**Backend**
- FastAPI (Python)
- scikit-learn (TF-IDF + cosine similarity)
- pandas / numpy

**Data**
- [MovieLens 32M Dataset](https://grouplens.org/datasets/movielens/) for movie genres, user tags, and optional aggregate ratings
- Streaming platform availability data via Kaggle

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

## How It Works

1. User searches for a title
2. Backend looks up the title in our dataset and extracts its feature vector (genre, cast, tags, etc.)
3. TF-IDF vectorization + cosine similarity is computed across all titles in the dataset
4. Top N most similar titles are returned alongside their streaming platform availability
5. Results are ranked by similarity score and displayed in the UI

---

## Project Structure

```
StreamCompass/
├── backend/
│   ├── main.py            # FastAPI app and routes
│   ├── recommender.py     # Recommendation engine logic
│   ├── data_loader.py     # Dataset loading and preprocessing
│   ├── requirements.txt
│   └── data/              # Dataset files
├── frontend/
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── pages/         # Page views
│   │   └── api/           # API call helpers
│   ├── index.html
│   └── package.json
└── README.md
```

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
