import os
import warnings

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

MOVIES_PATH = "ml/data/movies.csv"
RATINGS_PATH = "ml/data/ratings.csv"

N_COMPONENTS = 30

# Hybrid weights
CONTENT_WEIGHT = 0.35
COLLAB_WEIGHT = 0.50
POPULARITY_WEIGHT = 0.15

TOP_N = 10


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("Loading data...")

    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)

    print(f"Movies: {len(movies):,}")
    print(f"Ratings: {len(ratings):,}")

    return movies, ratings


# ============================================================
# CONTENT-BASED MODEL
# ============================================================

def build_content_model(movies):
    print("\nBuilding TF-IDF content model...")

    movies = movies.copy()

    # Replace missing genres
    movies["genres"] = movies["genres"].fillna("")

    # Replace separator with spaces
    movies["genres_clean"] = (
        movies["genres"]
        .str.replace("|", " ", regex=False)
    )

    # Combine title + genres
    movies["features"] = (
        movies["title"].fillna("")
        + " "
        + movies["genres_clean"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2
    )

    tfidf_matrix = vectorizer.fit_transform(
        movies["features"]
    )

    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    movie_to_idx = pd.Series(
        movies.index,
        index=movies["movieId"]
    ).to_dict()

    return movies, tfidf_matrix, movie_to_idx


# ============================================================
# COLLABORATIVE FILTERING MODEL
# ============================================================

def build_collaborative_model(ratings):
    print("\nBuilding collaborative model...")

    # Create user and movie indices
    user_ids = ratings["userId"].unique()
    movie_ids = ratings["movieId"].unique()

    user_to_idx = {
        user_id: idx
        for idx, user_id in enumerate(user_ids)
    }

    movie_to_idx = {
        movie_id: idx
        for idx, movie_id in enumerate(movie_ids)
    }

    rows = ratings["userId"].map(user_to_idx).values
    cols = ratings["movieId"].map(movie_to_idx).values
    data = ratings["rating"].values

    user_item_matrix = csr_matrix(
        (
            data,
            (rows, cols)
        ),
        shape=(
            len(user_ids),
            len(movie_ids)
        )
    )

    print(
        f"User-item matrix: "
        f"{user_item_matrix.shape}"
    )

    print(
        f"Training SVD with "
        f"{N_COMPONENTS} components..."
    )

    svd = TruncatedSVD(
        n_components=N_COMPONENTS,
        random_state=42
    )

    user_factors = svd.fit_transform(
        user_item_matrix
    )

    movie_factors = svd.components_

    explained_variance = svd.explained_variance_ratio_.sum()

    print(
        f"Explained variance: "
        f"{explained_variance:.4f}"
    )

    return (
        svd,
        user_item_matrix,
        user_factors,
        movie_factors,
        user_to_idx,
        movie_to_idx
    )


# ============================================================
# POPULARITY MODEL
# ============================================================

def calculate_popularity(ratings, movies):
    print("\nCalculating popularity...")

    statistics = (
        ratings
        .groupby("movieId")
        .agg(
            rating_count=("rating", "count"),
            average_rating=("rating", "mean")
        )
        .reset_index()
    )

    movies = movies.merge(
        statistics,
        on="movieId",
        how="left"
    )

    movies["rating_count"] = (
        movies["rating_count"]
        .fillna(0)
    )

    movies["average_rating"] = (
        movies["average_rating"]
        .fillna(0)
    )

    # Bayesian weighted rating
    C = ratings["rating"].mean()

    # Minimum number of ratings
    m = ratings["movieId"].value_counts().quantile(0.60)

    movies["weighted_rating"] = (
        (
            movies["rating_count"]
            /
            (movies["rating_count"] + m)
        )
        *
        movies["average_rating"]
        +
        (
            m
            /
            (movies["rating_count"] + m)
        )
        *
        C
    )

    # Normalize to [0, 1]
    min_score = movies["weighted_rating"].min()
    max_score = movies["weighted_rating"].max()

    if max_score > min_score:
        movies["popularity_score"] = (
            (
                movies["weighted_rating"]
                - min_score
            )
            /
            (max_score - min_score)
        )
    else:
        movies["popularity_score"] = 0

    return movies


# ============================================================
# CONTENT SCORES
# ============================================================

def get_content_scores(
    user_id,
    ratings,
    movies,
    tfidf_matrix,
    movie_to_idx
):
    """
    Build a user profile from movies the user rated
    and calculate cosine similarity against all movies.
    """

    user_ratings = ratings[
        ratings["userId"] == user_id
    ]

    if user_ratings.empty:
        return np.zeros(
            len(movies),
            dtype=float
        )

    profile_vectors = []
    weights = []

    for _, row in user_ratings.iterrows():

        movie_id = row["movieId"]

        if movie_id not in movie_to_idx:
            continue

        movie_index = movie_to_idx[movie_id]

        # Sparse vector -> ndarray
        vector = (
            tfidf_matrix[movie_index]
            .toarray()
            .ravel()
        )

        rating = float(row["rating"])

        # Center ratings around 3
        weight = rating - 3.0

        profile_vectors.append(vector)
        weights.append(weight)

    if not profile_vectors:
        return np.zeros(
            len(movies),
            dtype=float
        )

    profile_vectors = np.asarray(
        profile_vectors,
        dtype=np.float64
    )

    weights = np.asarray(
        weights,
        dtype=np.float64
    )

    # Avoid all-zero weights
    if np.allclose(weights, 0):
        weights = np.ones_like(weights)

    # Weighted profile
    user_profile = np.average(
        profile_vectors,
        axis=0,
        weights=np.abs(weights)
    )

    # Explicitly convert to ndarray
    user_profile = np.asarray(
        user_profile,
        dtype=np.float64
    ).reshape(1, -1)

    # IMPORTANT:
    # cosine_similarity must receive ndarray/sparse matrix,
    # never np.matrix.
    similarity = cosine_similarity(
        user_profile,
        tfidf_matrix
    )

    similarity = np.asarray(
        similarity
    ).ravel()

    # Convert cosine similarity from [-1, 1] to [0, 1]
    similarity = (
        similarity + 1
    ) / 2

    return similarity


# ============================================================
# COLLABORATIVE SCORES
# ============================================================

def get_collaborative_scores(
    user_id,
    user_to_idx,
    user_factors,
    movie_factors,
    movie_id_to_svd_idx,
    n_movies
):
    """
    Predict preference scores for all movies
    using the SVD latent representation.
    """

    if user_id not in user_to_idx:
        return np.zeros(
            n_movies,
            dtype=float
        )

    user_index = user_to_idx[user_id]

    user_vector = user_factors[user_index]

    # Predict scores for all movies represented
    # in collaborative model
    predicted = (
        user_vector
        @
        movie_factors
    )

    predicted = np.asarray(
        predicted
    ).ravel()

    # Normalize to [0, 1]
    min_score = predicted.min()
    max_score = predicted.max()

    if max_score > min_score:
        predicted = (
            predicted - min_score
        ) / (
            max_score - min_score
        )
    else:
        predicted = np.zeros_like(
            predicted
        )

    # Map SVD movie scores to original movie dataframe
    result = np.zeros(
        n_movies,
        dtype=float
    )

    for movie_id, movie_index in movie_id_to_svd_idx.items():

        if movie_id in movie_to_main_idx:
            main_index = movie_to_main_idx[movie_id]

            if movie_index < len(predicted):
                result[main_index] = (
                    predicted[movie_index]
                )

    return result


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_scores(scores):
    scores = np.asarray(
        scores,
        dtype=np.float64
    )

    min_score = np.min(scores)
    max_score = np.max(scores)

    if max_score > min_score:
        return (
            scores - min_score
        ) / (
            max_score - min_score
        )

    return np.zeros_like(scores)


# ============================================================
# HYBRID RECOMMENDATION
# ============================================================

def recommend(
    user_id,
    movies,
    ratings,
    tfidf_matrix,
    content_movie_to_idx,
    user_to_idx,
    user_factors,
    movie_factors,
    svd_movie_to_idx,
    top_n=10
):
    print(
        f"\nGenerating recommendations "
        f"for user {user_id}..."
    )

    # --------------------------------------------------------
    # Content scores
    # --------------------------------------------------------

    content_scores = get_content_scores(
        user_id=user_id,
        ratings=ratings,
        movies=movies,
        tfidf_matrix=tfidf_matrix,
        movie_to_idx=content_movie_to_idx
    )

    content_scores = normalize_scores(
        content_scores
    )

    # --------------------------------------------------------
    # Collaborative scores
    # --------------------------------------------------------

    collaborative_scores = (
        get_collaborative_scores(
            user_id=user_id,
            user_to_idx=user_to_idx,
            user_factors=user_factors,
            movie_factors=movie_factors,
            movie_id_to_svd_idx=svd_movie_to_idx,
            n_movies=len(movies)
        )
    )

    collaborative_scores = normalize_scores(
        collaborative_scores
    )

    # --------------------------------------------------------
    # Popularity
    # --------------------------------------------------------

    popularity_scores = (
        movies["popularity_score"]
        .fillna(0)
        .values
    )

    popularity_scores = normalize_scores(
        popularity_scores
    )

    # --------------------------------------------------------
    # Hybrid score
    # --------------------------------------------------------

    hybrid_scores = (
        CONTENT_WEIGHT
        * content_scores
        +
        COLLAB_WEIGHT
        * collaborative_scores
        +
        POPULARITY_WEIGHT
        * popularity_scores
    )

    result = movies.copy()

    result["content_score"] = (
        content_scores
    )

    result["collaborative_score"] = (
        collaborative_scores
    )

    result["popularity_score"] = (
        popularity_scores
    )

    result["hybrid_score"] = (
        hybrid_scores
    )

    # --------------------------------------------------------
    # Remove movies already rated by user
    # --------------------------------------------------------

    watched_movies = set(
        ratings.loc[
            ratings["userId"] == user_id,
            "movieId"
        ]
    )

    result = result[
        ~result["movieId"].isin(
            watched_movies
        )
    ]

    # Sort
    result = result.sort_values(
        "hybrid_score",
        ascending=False
    )

    return result.head(top_n)


# ============================================================
# MAIN
# ============================================================

def main():

    global movie_to_main_idx

    print("=" * 60)
    print("HYBRID MOVIE RECOMMENDATION SYSTEM")
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    movies, ratings = load_data()

    # --------------------------------------------------------
    # Content model
    # --------------------------------------------------------

    (
        movies,
        tfidf_matrix,
        content_movie_to_idx
    ) = build_content_model(
        movies
    )

    # Mapping from main dataframe
    movie_to_main_idx = {
        movie_id: idx
        for idx, movie_id
        in enumerate(movies["movieId"])
    }

    # --------------------------------------------------------
    # Collaborative model
    # --------------------------------------------------------

    (
        svd,
        user_item_matrix,
        user_factors,
        movie_factors,
        user_to_idx,
        svd_movie_to_idx
    ) = build_collaborative_model(
        ratings
    )

    # --------------------------------------------------------
    # Popularity
    # --------------------------------------------------------

    movies = calculate_popularity(
        ratings,
        movies
    )

    # --------------------------------------------------------
    # Example user
    # --------------------------------------------------------

    example_user = int(
        ratings["userId"].iloc[0]
    )

    # Better example:
    # select a user with at least 20 ratings
    user_counts = (
        ratings["userId"]
        .value_counts()
    )

    active_users = user_counts[
        user_counts >= 20
    ]

    if len(active_users) > 0:
        example_user = int(
            active_users.index[0]
        )

    print(
        f"\nExample User: {example_user}"
    )

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    recommendations = recommend(
        user_id=example_user,
        movies=movies,
        ratings=ratings,
        tfidf_matrix=tfidf_matrix,
        content_movie_to_idx=content_movie_to_idx,
        user_to_idx=user_to_idx,
        user_factors=user_factors,
        movie_factors=movie_factors,
        svd_movie_to_idx=svd_movie_to_idx,
        top_n=TOP_N
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("HYBRID RECOMMENDATIONS")
    print("=" * 60)

    display_columns = [
        "title",
        "genres",
        "content_score",
        "collaborative_score",
        "popularity_score",
        "hybrid_score"
    ]

    print(
        recommendations[
            display_columns
        ].to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 60)
    print("MODEL CONFIGURATION")
    print("=" * 60)

    print(
        f"Content weight: "
        f"{CONTENT_WEIGHT}"
    )

    print(
        f"Collaborative weight: "
        f"{COLLAB_WEIGHT}"
    )

    print(
        f"Popularity weight: "
        f"{POPULARITY_WEIGHT}"
    )

    print(
        f"SVD components: "
        f"{N_COMPONENTS}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()