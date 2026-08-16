import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

N_COMPONENTS = 30
CANDIDATE_SIZE = 200
TOP_N = 10

COLLAB_WEIGHT = 0.90
CONTENT_WEIGHT = 0.07
POPULARITY_WEIGHT = 0.03

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("Loading data...")

    movies = pd.read_csv("ml/data/movies.csv")
    ratings = pd.read_csv("ml/data/ratings.csv")

    print(f"Movies: {len(movies):,}")
    print(f"Ratings: {len(ratings):,}")

    return movies, ratings


# ============================================================
# CREATE TRAIN/TEST SPLIT
# ============================================================

def create_train_test(ratings, min_ratings=20):
    print("\nCreating train/test split...")

    user_counts = ratings["userId"].value_counts()

    valid_users = user_counts[user_counts >= min_ratings].index

    ratings = ratings[ratings["userId"].isin(valid_users)].copy()

    # Last rating of every eligible user goes to test
    ratings = ratings.sort_values(["userId", "timestamp"])

    test = ratings.groupby("userId").tail(1)

    train = ratings.drop(test.index)

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

    return matrix, user_ids, movie_ids, user_to_index, movie_to_index


# ============================================================
# TRAIN SVD
# ============================================================

def train_svd(matrix):
    print(f"\nTraining SVD with {N_COMPONENTS} components...")

    svd = TruncatedSVD(
        n_components=N_COMPONENTS,
        random_state=RANDOM_STATE
    )

    user_factors = svd.fit_transform(matrix)

    item_factors = svd.components_

    print(
        f"Explained variance: "
        f"{svd.explained_variance_ratio_.sum():.4f}"
    )

    print(f"User factor matrix: {user_factors.shape}")
    print(f"Item factor matrix: {item_factors.shape}")

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

    movies["title_clean"] = (
        movies["title"]
        .fillna("")
        .str.replace(r"\(\d{4}\)", "", regex=True)
    )

    movies["features"] = (
        movies["title_clean"]
        + " "
        + movies["genres"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(
        movies["features"]
    )

    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    return movies, vectorizer, tfidf_matrix


# ============================================================
# POPULARITY
# ============================================================

def calculate_popularity(ratings):
    print("\nCalculating popularity...")

    stats = (
        ratings.groupby("movieId")
        .agg(
            rating_count=("rating", "count"),
            average_rating=("rating", "mean")
        )
        .reset_index()
    )

    # Log transform prevents very popular movies
    # from completely dominating the score.
    stats["count_score"] = np.log1p(
        stats["rating_count"]
    )

    scaler = MinMaxScaler()

    stats["popularity_score"] = scaler.fit_transform(
        stats[["count_score"]]
    ).ravel()

    return stats


# ============================================================
# CONTENT SCORE
# ============================================================

def get_content_scores(
    user_id,
    train,
    movies,
    tfidf_matrix,
    movie_to_index
):
    user_ratings = train[
        train["userId"] == user_id
    ]

    if user_ratings.empty:
        return np.zeros(len(movies), dtype=np.float32)

    # Only use movies rated highly by the user.
    liked = user_ratings[
        user_ratings["rating"] >= 4.0
    ]

    if liked.empty:
        liked = user_ratings.nlargest(
            5,
            "rating"
        )

    movie_indices = []

    for movie_id in liked["movieId"]:
        if movie_id in movie_to_index:
            movie_indices.append(
                movie_to_index[movie_id]
            )

    if not movie_indices:
        return np.zeros(
            len(movies),
            dtype=np.float32
        )

    # We need movie IDs from the original movies dataframe.
    movie_id_to_row = {
        movie_id: index
        for index, movie_id in enumerate(
            movies["movieId"]
        )
    }

    content_vectors = []

    for movie_id in liked["movieId"]:
        if movie_id in movie_id_to_row:
            content_vectors.append(
                tfidf_matrix[
                    movie_id_to_row[movie_id]
                ]
            )

    if not content_vectors:
        return np.zeros(
            len(movies),
            dtype=np.float32
        )

    user_profile = sum(content_vectors)

    similarities = cosine_similarity(
        user_profile,
        tfidf_matrix
    ).ravel()

    return similarities.astype(np.float32)


# ============================================================
# COLLABORATIVE SCORES
# ============================================================

def get_collaborative_scores(
    user_id,
    user_to_index,
    user_factors,
    item_factors,
    movie_ids,
    movies
):
    scores = np.zeros(
        len(movies),
        dtype=np.float32
    )

    if user_id not in user_to_index:
        return scores

    user_index = user_to_index[user_id]

    user_vector = user_factors[user_index]

    predictions = (
        user_vector @ item_factors
    )

    movie_id_to_score = {
        movie_id: score
        for movie_id, score in zip(
            movie_ids,
            predictions
        )
    }

    for i, movie_id in enumerate(
        movies["movieId"]
    ):
        if movie_id in movie_id_to_score:
            scores[i] = movie_id_to_score[movie_id]

    return scores


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_scores(scores):
    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:
        return np.zeros_like(scores)

    return (
        (scores - minimum)
        / (maximum - minimum)
    )


# ============================================================
# HYBRID RECOMMENDATION
# ============================================================

def recommend(
    user_id,
    train,
    movies,
    tfidf_matrix,
    user_to_index,
    user_factors,
    item_factors,
    movie_ids,
    popularity_stats
):
    print(
        f"\nGenerating recommendations "
        f"for user {user_id}..."
    )

    # --------------------------------------------------------
    # 1. Collaborative scores
    # --------------------------------------------------------

    collaborative_scores = get_collaborative_scores(
        user_id=user_id,
        user_to_index=user_to_index,
        user_factors=user_factors,
        item_factors=item_factors,
        movie_ids=movie_ids,
        movies=movies
    )

    collaborative_scores = normalize_scores(
        collaborative_scores
    )

    # --------------------------------------------------------
    # 2. Get candidate movies
    # --------------------------------------------------------

    rated_movies = set(
        train.loc[
            train["userId"] == user_id,
            "movieId"
        ]
    )

    candidate_indices = np.argsort(
        collaborative_scores
    )[::-1]

    candidates = []

    for index in candidate_indices:

        movie_id = movies.iloc[index]["movieId"]

        if movie_id in rated_movies:
            continue

        candidates.append(index)

        if len(candidates) >= CANDIDATE_SIZE:
            break

    candidates = np.array(
        candidates,
        dtype=int
    )

    # --------------------------------------------------------
    # 3. Content scores
    # --------------------------------------------------------

    content_scores = get_content_scores(
        user_id=user_id,
        train=train,
        movies=movies,
        tfidf_matrix=tfidf_matrix,
        movie_to_index={
            movie_id: i
            for i, movie_id in enumerate(
                movie_ids
            )
        }
    )

    content_scores = normalize_scores(
        content_scores
    )

    # --------------------------------------------------------
    # 4. Popularity
    # --------------------------------------------------------

    popularity_map = dict(
        zip(
            popularity_stats["movieId"],
            popularity_stats["popularity_score"]
        )
    )

    popularity_scores = np.array(
        [
            popularity_map.get(
                movie_id,
                0.0
            )
            for movie_id in movies["movieId"]
        ],
        dtype=np.float32
    )

    popularity_scores = normalize_scores(
        popularity_scores
    )

    # --------------------------------------------------------
    # 5. Re-ranking
    # --------------------------------------------------------

    final_scores = (
        COLLAB_WEIGHT
        * collaborative_scores
        +
        CONTENT_WEIGHT
        * content_scores
        +
        POPULARITY_WEIGHT
        * popularity_scores
    )

    candidate_scores = final_scores[
        candidates
    ]

    ranked_positions = np.argsort(
        candidate_scores
    )[::-1]

    selected = candidates[
        ranked_positions[:TOP_N]
    ]

    result = movies.iloc[
        selected
    ].copy()

    result["collaborative_score"] = (
        collaborative_scores[selected]
    )

    result["content_score"] = (
        content_scores[selected]
    )

    result["popularity_score"] = (
        popularity_scores[selected]
    )

    result["hybrid_score"] = (
        final_scores[selected]
    )

    return result[
        [
            "title",
            "genres",
            "collaborative_score",
            "content_score",
            "popularity_score",
            "hybrid_score"
        ]
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("HYBRID MOVIE RECOMMENDATION SYSTEM V3")
    print("=" * 60)

    # Load
    movies, ratings = load_data()

    # Train/test
    train, test = create_train_test(
        ratings
    )

    # User-item matrix
    (
        matrix,
        user_ids,
        movie_ids,
        user_to_index,
        movie_to_index
    ) = create_user_item_matrix(
        train
    )

    # SVD
    (
        svd,
        user_factors,
        item_factors
    ) = train_svd(
        matrix
    )

    # Content model
    (
        movies,
        vectorizer,
        tfidf_matrix
    ) = build_content_model(
        movies
    )

    # Popularity
    popularity_stats = calculate_popularity(
        train
    )

    # Example user
    example_user = int(
        user_ids[0]
    )

    print(
        f"\nExample User: {example_user}"
    )

    recommendations = recommend(
        user_id=example_user,
        train=train,
        movies=movies,
        tfidf_matrix=tfidf_matrix,
        user_to_index=user_to_index,
        user_factors=user_factors,
        item_factors=item_factors,
        movie_ids=movie_ids,
        popularity_stats=popularity_stats
    )

    print("\n")
    print("=" * 60)
    print("HYBRID V3 RECOMMENDATIONS")
    print("=" * 60)

    print(
        recommendations.to_string(
            index=False
        )
    )

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


if __name__ == "__main__":
    main()