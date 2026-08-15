import os
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "ml/data"

MOVIES_FILE = os.path.join(DATA_DIR, "movies.csv")
RATINGS_FILE = os.path.join(DATA_DIR, "ratings.csv")

N_COMPONENTS = 30
N_RECOMMENDATIONS = 10

# Collaborative is the main model.
COLLAB_WEIGHT = 0.80
CONTENT_WEIGHT = 0.15
POPULARITY_WEIGHT = 0.05

MIN_RATINGS_PER_USER = 20

# Number of movies used for re-ranking.
# We do NOT calculate scores for all 86k movies.
CANDIDATE_SIZE = 200


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("Loading data...")

    movies = pd.read_csv(MOVIES_FILE)
    ratings = pd.read_csv(RATINGS_FILE)

    print(f"Movies: {len(movies):,}")
    print(f"Ratings: {len(ratings):,}")

    return movies, ratings


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def create_train_test(ratings):
    print("\nCreating train/test split...")

    user_counts = ratings.groupby("userId").size()

    valid_users = user_counts[
        user_counts >= MIN_RATINGS_PER_USER
    ].index

    ratings_filtered = ratings[
        ratings["userId"].isin(valid_users)
    ].copy()

    # Last rating of every eligible user becomes test data.
    ratings_filtered = ratings_filtered.sort_values(
        ["userId", "timestamp"]
    )

    test = ratings_filtered.groupby(
        "userId",
        group_keys=False
    ).tail(1)

    train = ratings_filtered.drop(
        test.index
    )

    print(f"Users: {len(valid_users):,}")
    print(f"Training ratings: {len(train):,}")
    print(f"Test ratings: {len(test):,}")

    return train, test


# ============================================================
# USER-ITEM MATRIX
# ============================================================

def create_user_item_matrix(train):
    print("\nCreating User-Item matrix...")

    user_ids = train["userId"].unique()
    movie_ids = train["movieId"].unique()

    user_to_index = {
        user_id: index
        for index, user_id in enumerate(user_ids)
    }

    movie_to_index = {
        movie_id: index
        for index, movie_id in enumerate(movie_ids)
    }

    rows = train["userId"].map(user_to_index).values
    cols = train["movieId"].map(movie_to_index).values
    values = train["rating"].values.astype(np.float32)

    matrix = csr_matrix(
        (values, (rows, cols)),
        shape=(len(user_ids), len(movie_ids)),
        dtype=np.float32
    )

    print(f"Matrix shape: {matrix.shape}")
    print(f"Non-zero ratings: {matrix.nnz:,}")

    return (
        matrix,
        user_ids,
        movie_ids,
        user_to_index,
        movie_to_index
    )


# ============================================================
# TRAIN SVD
# ============================================================

def train_svd(user_item_matrix):
    print(
        f"\nTraining SVD with "
        f"{N_COMPONENTS} components..."
    )

    svd = TruncatedSVD(
        n_components=N_COMPONENTS,
        random_state=42
    )

    user_factors = svd.fit_transform(
        user_item_matrix
    )

    item_factors = svd.components_

    explained_variance = (
        svd.explained_variance_ratio_.sum()
    )

    print(
        f"Explained variance: "
        f"{explained_variance:.4f}"
    )

    print(
        f"User factor matrix: "
        f"{user_factors.shape}"
    )

    print(
        f"Item factor matrix: "
        f"{item_factors.shape}"
    )

    return svd, user_factors, item_factors


# ============================================================
# TF-IDF CONTENT MODEL
# ============================================================

def build_content_model(movies):
    print("\nBuilding TF-IDF content model...")

    movies = movies.copy()

    movies["genres"] = (
        movies["genres"]
        .fillna("")
        .str.replace("|", " ", regex=False)
    )

    movies["title"] = (
        movies["title"]
        .fillna("")
    )

    movies["features"] = (
        movies["title"] + " "
        + movies["genres"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000
    )

    tfidf_matrix = vectorizer.fit_transform(
        movies["features"]
    )

    print(
        f"TF-IDF matrix shape: "
        f"{tfidf_matrix.shape}"
    )

    return movies, vectorizer, tfidf_matrix


# ============================================================
# POPULARITY
# ============================================================

def calculate_popularity(train):
    print("\nCalculating popularity...")

    stats = (
        train.groupby("movieId")
        .agg(
            rating_count=("rating", "count"),
            average_rating=("rating", "mean")
        )
        .reset_index()
    )

    # Log scaling prevents movies with huge rating counts
    # from completely dominating popularity.
    stats["popularity_score"] = (
        np.log1p(stats["rating_count"])
        / np.log1p(stats["rating_count"].max())
    )

    return stats


# ============================================================
# NORMALIZATION
# ============================================================

def min_max_normalize(values):
    values = np.asarray(
        values,
        dtype=np.float32
    )

    if len(values) == 0:
        return values

    minimum = values.min()
    maximum = values.max()

    if maximum - minimum < 1e-8:
        return np.ones_like(values)

    return (
        (values - minimum)
        / (maximum - minimum)
    )


# ============================================================
# CONTENT SCORE
# ============================================================

def calculate_content_scores(
    user_movie_ids,
    candidate_movie_ids,
    movies,
    tfidf_matrix,
    movie_id_to_row
):
    """
    Calculate content similarity between the user's
    previously highly-rated movies and candidates.

    Only candidate movies are processed, avoiding
    a huge 86k x 86k similarity matrix.
    """

    liked_movie_ids = []

    # We need ratings here, so the caller supplies
    # movies that the user has interacted with.
    for movie_id in user_movie_ids:
        if movie_id in movie_id_to_row:
            liked_movie_ids.append(movie_id)

    if not liked_movie_ids:
        return np.zeros(
            len(candidate_movie_ids),
            dtype=np.float32
        )

    liked_indices = [
        movie_id_to_row[movie_id]
        for movie_id in liked_movie_ids
    ]

    candidate_indices = [
        movie_id_to_row[movie_id]
        for movie_id in candidate_movie_ids
    ]

    user_profile = (
        tfidf_matrix[liked_indices]
        .mean(axis=0)
    )

    # Convert np.matrix -> ndarray.
    user_profile = np.asarray(
        user_profile
    )

    candidate_vectors = (
        tfidf_matrix[candidate_indices]
    )

    # Since TF-IDF vectors are normalized,
    # dot product is cosine similarity.
    scores = (
        candidate_vectors @ user_profile.T
    )

    scores = np.asarray(
        scores
    ).ravel()

    return scores.astype(
        np.float32
    )


# ============================================================
# RECOMMEND
# ============================================================

def recommend(
    user_id,
    train,
    movies,
    tfidf_matrix,
    user_factors,
    item_factors,
    user_to_index,
    movie_to_index,
    movie_id_to_row,
    popularity_stats,
    n=N_RECOMMENDATIONS
):
    if user_id not in user_to_index:
        return pd.DataFrame()

    user_index = user_to_index[user_id]

    # --------------------------------------------------------
    # Movies already rated by the user
    # --------------------------------------------------------

    user_ratings = train[
        train["userId"] == user_id
    ]

    rated_movie_ids = set(
        user_ratings["movieId"]
    )

    # Highly-rated movies for content profile.
    liked_movie_ids = list(
        user_ratings[
            user_ratings["rating"] >= 4.0
        ]["movieId"]
    )

    # --------------------------------------------------------
    # Collaborative scores
    # --------------------------------------------------------

    user_vector = user_factors[
        user_index
    ]

    collaborative_scores = (
        user_vector @ item_factors
    )

    collaborative_scores = (
        collaborative_scores.astype(
            np.float32
        )
    )

    # Normalize collaborative scores.
    collaborative_scores = (
        min_max_normalize(
            collaborative_scores
        )
    )

    # --------------------------------------------------------
    # Candidate generation
    # --------------------------------------------------------

    available_movie_ids = np.array(
        list(movie_to_index.keys())
    )

    # Remove already rated movies.
    candidate_mask = np.array([
        movie_id not in rated_movie_ids
        for movie_id in available_movie_ids
    ])

    candidate_movie_ids = (
        available_movie_ids[candidate_mask]
    )

    candidate_indices = np.array([
        movie_to_index[movie_id]
        for movie_id in candidate_movie_ids
    ])

    candidate_collab_scores = (
        collaborative_scores[candidate_indices]
    )

    # Select only top collaborative candidates.
    if len(candidate_indices) > CANDIDATE_SIZE:

        top_positions = np.argpartition(
            candidate_collab_scores,
            -CANDIDATE_SIZE
        )[-CANDIDATE_SIZE:]

        candidate_movie_ids = (
            candidate_movie_ids[top_positions]
        )

        candidate_indices = (
            candidate_indices[top_positions]
        )

        candidate_collab_scores = (
            candidate_collab_scores[
                top_positions
            ]
        )

    # --------------------------------------------------------
    # Content scores
    # --------------------------------------------------------

    content_scores = calculate_content_scores(
        liked_movie_ids,
        candidate_movie_ids,
        movies,
        tfidf_matrix,
        movie_id_to_row
    )

    content_scores = min_max_normalize(
        content_scores
    )

    # --------------------------------------------------------
    # Popularity scores
    # --------------------------------------------------------

    popularity_map = dict(
        zip(
            popularity_stats["movieId"],
            popularity_stats[
                "popularity_score"
            ]
        )
    )

    popularity_scores = np.array([
        popularity_map.get(
            movie_id,
            0.0
        )
        for movie_id in candidate_movie_ids
    ], dtype=np.float32)

    popularity_scores = min_max_normalize(
        popularity_scores
    )

    # --------------------------------------------------------
    # HYBRID SCORE
    # --------------------------------------------------------

    hybrid_scores = (
        COLLAB_WEIGHT
        * candidate_collab_scores
        +
        CONTENT_WEIGHT
        * content_scores
        +
        POPULARITY_WEIGHT
        * popularity_scores
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = pd.DataFrame({
        "movieId": candidate_movie_ids,
        "collaborative_score":
            candidate_collab_scores,
        "content_score":
            content_scores,
        "popularity_score":
            popularity_scores,
        "hybrid_score":
            hybrid_scores
    })

    result = result.merge(
        movies[
            ["movieId", "title", "genres"]
        ],
        on="movieId",
        how="left"
    )

    result = result.sort_values(
        "hybrid_score",
        ascending=False
    )

    return result.head(n)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("HYBRID MOVIE RECOMMENDATION SYSTEM V2")
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    movies, ratings = load_data()

    # --------------------------------------------------------
    # Train/Test
    # --------------------------------------------------------

    train, test = create_train_test(
        ratings
    )

    # --------------------------------------------------------
    # User-item matrix
    # --------------------------------------------------------

    (
        user_item_matrix,
        user_ids,
        movie_ids,
        user_to_index,
        movie_to_index
    ) = create_user_item_matrix(
        train
    )

    # --------------------------------------------------------
    # SVD
    # --------------------------------------------------------

    (
        svd,
        user_factors,
        item_factors
    ) = train_svd(
        user_item_matrix
    )

    # --------------------------------------------------------
    # Content model
    # --------------------------------------------------------

    (
        movies,
        vectorizer,
        tfidf_matrix
    ) = build_content_model(
        movies
    )

    movie_id_to_row = {
        movie_id: index
        for index, movie_id
        in enumerate(
            movies["movieId"]
        )
    }

    # --------------------------------------------------------
    # Popularity
    # --------------------------------------------------------

    popularity_stats = calculate_popularity(
        train
    )

    # --------------------------------------------------------
    # Example user
    # --------------------------------------------------------

    example_user = int(
        test["userId"].iloc[0]
    )

    print(
        f"\nExample User: "
        f"{example_user}"
    )

    print(
        f"\nGenerating recommendations "
        f"for user {example_user}..."
    )

    recommendations = recommend(
        user_id=example_user,
        train=train,
        movies=movies,
        tfidf_matrix=tfidf_matrix,
        user_factors=user_factors,
        item_factors=item_factors,
        user_to_index=user_to_index,
        movie_to_index=movie_to_index,
        movie_id_to_row=movie_id_to_row,
        popularity_stats=popularity_stats,
        n=N_RECOMMENDATIONS
    )

    print("\n")
    print("=" * 60)
    print("HYBRID V2 RECOMMENDATIONS")
    print("=" * 60)

    if recommendations.empty:
        print("No recommendations available.")
    else:
        display_columns = [
            "title",
            "genres",
            "collaborative_score",
            "content_score",
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

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("MODEL CONFIGURATION")
    print("=" * 60)

    print(
        f"Collaborative weight: "
        f"{COLLAB_WEIGHT}"
    )

    print(
        f"Content weight: "
        f"{CONTENT_WEIGHT}"
    )

    print(
        f"Popularity weight: "
        f"{POPULARITY_WEIGHT}"
    )

    print(
        f"SVD components: "
        f"{N_COMPONENTS}"
    )

    print(
        f"Candidate pool: "
        f"{CANDIDATE_SIZE}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()