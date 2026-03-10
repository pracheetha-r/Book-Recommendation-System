from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


class RecommendRequest(BaseModel):
    title: str
    n: int = 5


class BookOut(BaseModel):
    title: str
    authors: str


class CombinedRecommendations(BaseModel):
    content_based: List[BookOut]
    collaborative: List[BookOut]


app = FastAPI(title="Book Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_lock = threading.Lock()
_model_ready = False
_model_error: Optional[str] = None

# Filled when ready
books_with_tags = None
cosine_sim = None
indices = None
book_user_matrix = None
book_titles_indices = None
model_knn = None


def _build_recommender():
    books = pd.read_csv("books.csv")
    book_tags = pd.read_csv("book_tags.csv")
    tags = pd.read_csv("tags.csv")

    book_tags = book_tags.merge(tags, on="tag_id", how="left")

    books_tags_joined = books.merge(book_tags, on="goodreads_book_id", how="left")

    books_with_tags = (
        books_tags_joined.groupby(["book_id", "title", "authors"])["tag_name"]
        .apply(lambda x: " ".join(x.dropna()))
        .reset_index()
    )

    books_with_tags["features"] = (
        books_with_tags["title"]
        + " "
        + books_with_tags["authors"]
        + " "
        + books_with_tags["tag_name"]
    )

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(books_with_tags["features"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(books_with_tags.index, books_with_tags["title"]).drop_duplicates()

    # collaborative filtering preparation
    ratings = pd.read_csv("ratings.csv")
    book_rating_counts = ratings["book_id"].value_counts()
    popular_books = book_rating_counts[book_rating_counts >= 50].index
    ratings_filtered = ratings[ratings["book_id"].isin(popular_books)]

    user_rating_counts = ratings["user_id"].value_counts()
    active_users = user_rating_counts[user_rating_counts >= 50].index
    ratings_filtered = ratings_filtered[ratings_filtered["user_id"].isin(active_users)]

    ratings_with_titles = ratings_filtered.merge(
        books[["book_id", "title"]], on="book_id"
    )
    book_user_matrix = (
        ratings_with_titles.pivot_table(
            index="title", columns="user_id", values="rating"
        ).fillna(0)
    )
    book_user_sparse = csr_matrix(book_user_matrix.values)
    book_titles_indices = list(book_user_matrix.index)

    model_knn = NearestNeighbors(metric="cosine", algorithm="brute")
    model_knn.fit(book_user_sparse)

    return (
        books_with_tags,
        cosine_sim,
        indices,
        book_user_matrix,
        book_titles_indices,
        model_knn,
    )

def _load_models_background():
    global _model_ready, _model_error
    global books_with_tags, cosine_sim, indices, book_user_matrix, book_titles_indices, model_knn

    try:
        built = _build_recommender()
        with _model_lock:
            (
                books_with_tags,
                cosine_sim,
                indices,
                book_user_matrix,
                book_titles_indices,
                model_knn,
            ) = built
            _model_ready = True
            _model_error = None
    except Exception as exc:
        with _model_lock:
            _model_ready = False
            _model_error = f"{type(exc).__name__}: {exc}"


@app.on_event("startup")
def _startup():
    # Load heavy models in a background thread so the server can accept connections immediately.
    t = threading.Thread(target=_load_models_background, daemon=True)
    t.start()


@app.get("/health")
def health():
    with _model_lock:
        return {"ok": True, "ready": _model_ready, "error": _model_error}


def recommend_books(title: str, n: int = 5) -> List[BookOut]:
    with _model_lock:
        if not _model_ready:
            raise HTTPException(status_code=503, detail="Model is still loading")
    if title not in indices:
        raise KeyError("Book not found in Dataset")

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : n + 1]
    book_indices = [i[0] for i in sim_scores]

    recs = books_with_tags[["title", "authors"]].iloc[book_indices]
    return [BookOut(title=row.title, authors=row.authors) for _, row in recs.iterrows()]


def recommend_collaborative(title: str, n: int = 5) -> List[BookOut]:
    with _model_lock:
        if not _model_ready:
            raise HTTPException(status_code=503, detail="Model is still loading")
    if title not in book_titles_indices:
        raise KeyError("Book not found in filtered dataset")

    book_idx = book_titles_indices.index(title)
    distances, neighbor_indices = model_knn.kneighbors(
        book_user_matrix.iloc[book_idx, :].values.reshape(1, -1),
        n_neighbors=n + 1,
    )
    neighbor_idx = neighbor_indices.flatten()[1:]
    recommended_titles = [book_titles_indices[i] for i in neighbor_idx]

    recommendations = books_with_tags[books_with_tags["title"].isin(recommended_titles)]
    recommendations = (
        recommendations.set_index("title")
        .reindex(recommended_titles)
        .reset_index()
    )
    return [
        BookOut(title=row.title, authors=row.authors)
        for _, row in recommendations[["title", "authors"]].iterrows()
    ]


@app.post("/recommend", response_model=CombinedRecommendations)
def recommend_endpoint(payload: RecommendRequest):
    try:
        if payload.n <= 0:
            raise HTTPException(status_code=400, detail="n must be positive")
        content_recs = recommend_books(payload.title, payload.n)
        try:
            collab_recs = recommend_collaborative(payload.title, payload.n)
        except KeyError:
            collab_recs = []
        return CombinedRecommendations(
            content_based=content_recs,
            collaborative=collab_recs,
        )
    except KeyError as exc:
        # original content-based not found
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/titles", response_model=List[str])
def suggest_titles(query: str = Query("", alias="q"), limit: int = 10) -> List[str]:
    """
    Lightweight title suggestion endpoint for autocomplete.
    Performs a simple case-insensitive containment filter on book titles.
    """
    with _model_lock:
        if not _model_ready:
            return []
    q = query.strip().lower()
    if not q:
        return []

    # filter titles containing the query, preserve original casing
    mask = books_with_tags["title"].str.lower().str.contains(q, na=False)
    matches = books_with_tags.loc[mask, "title"].drop_duplicates().head(limit)
    return matches.tolist()
