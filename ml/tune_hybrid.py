import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CONFIGURATION
# ============================================================

N_USERS = 500
TOP_K = 10

SVD_COMPONENTS = 30
MAX_TFIDF_FEATURES = 20000

# Weight combinations to test
WEIGHT_COMBINATIONS = [
    (0.10, 0.80, 0.10),
    (0.15, 0.80, 0.05),
    (0.20, 0.75, 0.05),
    (0.20, 0.70, 0.10),
    (0.10, 0.85, 0.05),
    (0.25, 0.70, 0.05),
    (0.15, 0.75, 0.10),
]


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
# TRAIN / TEST SPLIT
# ============================================================

def create_train_test(ratings):

    print("\nCreating train/test split...")

    user_counts = ratings["userId"].value_counts()

    eligible_users = user_counts[user_counts >= 20].index

    ratings = ratings[
        ratings["userId"].isin(eligible_users)
    ].copy()

    # Last rating of each user -> test
    ratings = ratings.sort_values(
        ["userId", "timestamp"]
    )

    test = ratings.groupby(
        "userId",
        group_keys=False
    ).tail(1)

    train = ratings.drop(test.index)

    print(f"Users: {len(eligible_users):,}")
    print(f"Training ratings: {len(train):,}")
    print(f"Test ratings: {len(test):,}")

    return train, test


# ============================================================
# COLLABORATIVE MODEL
# ============================================================

def build_collaborative_model(train):

    print("\nCreating User-Item matrix...")

    user_ids = train["userId"].unique()
    movie_ids = train["movieId"].unique()

    user_to_index = {
        user_id: i
        for i, user_id in enumerate(user_ids)
    }

    movie_to_index = {
        movie_id: i
        for i, movie_id in enumerate(movie_ids)
    }

    rows = train["userId"].map(user_to_index)
    cols = train["movieId"].map(movie_to_index)

    matrix = csr_matrix(
        (
            train["rating"].values,
            (rows, cols)
        ),
        shape=(len(user_ids), len(movie_ids))
    )

    print(f"Matrix shape: {matrix.shape}")
    print(f"Non-zero ratings: {matrix.nnz:,}")

    print(
        f"\nTraining SVD with "
        f"{SVD_COMPONENTS} components..."
    )

    svd = TruncatedSVD(
        n_components=SVD_COMPONENTS,
        random_state=42
    )

    user_factors = svd.fit_transform(matrix)

    item_factors = svd.components_

    print(
        f"Explained variance: "
        f"{svd.explained_variance_ratio_.sum():.4f}"
    )

    print(
        f"User factor matrix: "
        f"{user_factors.shape}"
    )

    print(
        f"Item factor matrix: "
        f"{item_factors.shape}"
    )

    return (
        user_ids,
        movie_ids,
        user_to_index,
        movie_to_index,
        user_factors,
        item_factors
    )


# ============================================================
# CONTENT MODEL
# ============================================================

def build_content_model(movies):

    print("\nBuilding TF-IDF content model...")

    movies = movies.copy()

    movies["genres_clean"] = (
        movies["genres"]
        .fillna("")
        .str.replace("|", " ", regex=False)
    )

    movies["features"] = (
        movies["title"].fillna("")
        + " "
        + movies["genres_clean"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=MAX_TFIDF_FEATURES
    )

    tfidf_matrix = vectorizer.fit_transform(
        movies["features"]
    )

    movie_to_content_index = {
        movie_id: i
        for i, movie_id in enumerate(
            movies["movieId"]
        )
    }

    print(
        f"TF-IDF matrix shape: "
        f"{tfidf_matrix.shape}"
    )

    return (
        movies,
        tfidf_matrix,
        movie_to_content_index
    )


# ============================================================
# POPULARITY
# ============================================================

def calculate_popularity(train):

    print("\nCalculating popularity...")

    stats = train.groupby("movieId").agg(
        rating_count=("rating", "count"),
        average_rating=("rating", "mean")
    ).reset_index()

    # Log transformation prevents extremely popular
    # movies from dominating everything.
    stats["popularity_score"] = np.log1p(
        stats["rating_count"]
    )

    min_score = stats["popularity_score"].min()
    max_score = stats["popularity_score"].max()

    if max_score > min_score:

        stats["popularity_score"] = (
            stats["popularity_score"] - min_score
        ) / (max_score - min_score)

    else:

        stats["popularity_score"] = 0.0

    return stats


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_scores(scores):

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    min_value = scores.min()
    max_value = scores.max()

    if max_value == min_value:
        return np.zeros_like(scores)

    return (
        (scores - min_value)
        / (max_value - min_value)
    )


# ============================================================
# CONTENT SCORES
# ============================================================

def get_content_scores(
    user_ratings,
    movies,
    tfidf_matrix,
    movie_to_content_index
):

    rated_movies = []

    for movie_id in user_ratings:

        if movie_id in movie_to_content_index:

            rated_movies.append(
                movie_to_content_index[movie_id]
            )

    if not rated_movies:

        return np.zeros(
            tfidf_matrix.shape[0],
            dtype=np.float32
        )

    # Average profile of movies the user has rated
    user_profile = tfidf_matrix[
        rated_movies
    ].mean(axis=0)

    user_profile = np.asarray(
        user_profile
    )

    # Since TF-IDF vectors are normalized,
    # dot product gives cosine similarity.
    scores = tfidf_matrix.dot(
        user_profile.T
    )

    scores = np.asarray(
        scores
    ).ravel()

    return scores


# ============================================================
# COLLABORATIVE SCORES
# ============================================================

def get_collaborative_scores(
    user_id,
    user_to_index,
    movie_ids,
    movie_to_index,
    user_factors,
    item_factors
):

    scores = np.zeros(
        len(movie_ids),
        dtype=np.float32
    )

    if user_id not in user_to_index:
        return scores

    user_index = user_to_index[user_id]

    user_vector = user_factors[
        user_index
    ]

    # Only calculate scores for items that exist
    # in the SVD model.
    predicted = user_vector @ item_factors

    for movie_id, movie_index in movie_to_index.items():

        scores[movie_index] = predicted[movie_index]

    return scores


# ============================================================
# RECOMMENDATION
# ============================================================

def recommend(
    user_id,
    user_ratings,
    movies,
    tfidf_matrix,
    movie_to_content_index,
    user_to_index,
    movie_ids,
    movie_to_index,
    user_factors,
    item_factors,
    popularity_stats,
    content_weight,
    collaborative_weight,
    popularity_weight
):

    # --------------------------------------------------------
    # CONTENT
    # --------------------------------------------------------

    content_scores = get_content_scores(
        user_ratings,
        movies,
        tfidf_matrix,
        movie_to_content_index
    )

    # --------------------------------------------------------
    # COLLABORATIVE
    # --------------------------------------------------------

    collaborative_raw = get_collaborative_scores(
        user_id,
        user_to_index,
        movie_ids,
        movie_to_index,
        user_factors,
        item_factors
    )

    # --------------------------------------------------------
    # Align collaborative scores to all movies
    # --------------------------------------------------------

    collaborative_scores = np.zeros(
        len(movies),
        dtype=np.float32
    )

    movie_index_lookup = {
        movie_id: i
        for i, movie_id in enumerate(movie_ids)
    }

    for i, movie_id in enumerate(
        movies["movieId"]
    ):

        if movie_id in movie_index_lookup:

            collaborative_scores[i] = (
                collaborative_raw[
                    movie_index_lookup[movie_id]
                ]
            )

    # --------------------------------------------------------
    # POPULARITY
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

    # --------------------------------------------------------
    # NORMALIZE EACH MODEL
    # --------------------------------------------------------

    content_scores = normalize_scores(
        content_scores
    )

    collaborative_scores = normalize_scores(
        collaborative_scores
    )

    popularity_scores = normalize_scores(
        popularity_scores
    )

    # --------------------------------------------------------
    # HYBRID SCORE
    # --------------------------------------------------------

    hybrid_scores = (
        content_weight * content_scores
        +
        collaborative_weight * collaborative_scores
        +
        popularity_weight * popularity_scores
    )

    # --------------------------------------------------------
    # Remove movies already rated
    # --------------------------------------------------------

    rated_set = set(user_ratings)

    for i, movie_id in enumerate(
        movies["movieId"]
    ):

        if movie_id in rated_set:

            hybrid_scores[i] = -np.inf

    # --------------------------------------------------------
    # Top K
    # --------------------------------------------------------

    top_indices = np.argpartition(
        hybrid_scores,
        -TOP_K
    )[-TOP_K:]

    top_indices = top_indices[
        np.argsort(
            hybrid_scores[top_indices]
        )[::-1]
    ]

    return movies.iloc[
        top_indices
    ]["movieId"].tolist()


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    recommendations,
    test_movie
):

    recommended_set = set(
        recommendations
    )

    hit = (
        1
        if test_movie in recommended_set
        else 0
    )

    precision = hit / TOP_K
    recall = hit

    if hit:

        rank = recommendations.index(
            test_movie
        ) + 1

        ndcg = 1 / np.log2(
            rank + 1
        )

    else:

        ndcg = 0.0

    return (
        precision,
        recall,
        ndcg
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_weights(
    weights,
    users_to_evaluate,
    train,
    test,
    movies,
    tfidf_matrix,
    movie_to_content_index,
    user_to_index,
    movie_ids,
    movie_to_index,
    user_factors,
    item_factors,
    popularity_stats
):

    content_weight, collaborative_weight, popularity_weight = weights

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    # User -> movies rated in training
    user_history = (
        train.groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )

    test_lookup = dict(
        zip(
            test["userId"],
            test["movieId"]
        )
    )

    for user_id in users_to_evaluate:

        if user_id not in test_lookup:
            continue

        test_movie = test_lookup[user_id]

        history = user_history.get(
            user_id,
            set()
        )

        recommendations = recommend(
            user_id,
            history,
            movies,
            tfidf_matrix,
            movie_to_content_index,
            user_to_index,
            movie_ids,
            movie_to_index,
            user_factors,
            item_factors,
            popularity_stats,
            content_weight,
            collaborative_weight,
            popularity_weight
        )

        precision, recall, ndcg = calculate_metrics(
            recommendations,
            test_movie
        )

        precision_scores.append(
            precision
        )

        recall_scores.append(
            recall
        )

        ndcg_scores.append(
            ndcg
        )

    return (
        np.mean(precision_scores),
        np.mean(recall_scores),
        np.mean(ndcg_scores)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("HYBRID MODEL - WEIGHT TUNING")
    print("=" * 60)

    movies, ratings = load_data()

    train, test = create_train_test(
        ratings
    )

    (
        user_ids,
        movie_ids,
        user_to_index,
        movie_to_index,
        user_factors,
        item_factors
    ) = build_collaborative_model(
        train
    )

    (
        movies,
        tfidf_matrix,
        movie_to_content_index
    ) = build_content_model(
        movies
    )

    popularity_stats = calculate_popularity(
        train
    )

    # Use the same 500-user evaluation size
    # to keep runtime reasonable.
    available_users = list(
        test["userId"].unique()
    )

    np.random.seed(42)

    if len(available_users) > N_USERS:

        users_to_evaluate = list(
            np.random.choice(
                available_users,
                N_USERS,
                replace=False
            )
        )

    else:

        users_to_evaluate = available_users

    print(
        f"\nUsers evaluated per configuration: "
        f"{len(users_to_evaluate)}"
    )

    results = []

    for index, weights in enumerate(
        WEIGHT_COMBINATIONS,
        start=1
    ):

        print("\n" + "-" * 60)

        print(
            f"Configuration {index}/"
            f"{len(WEIGHT_COMBINATIONS)}"
        )

        print(
            f"Content:       {weights[0]:.2f}"
        )

        print(
            f"Collaborative: {weights[1]:.2f}"
        )

        print(
            f"Popularity:    {weights[2]:.2f}"
        )

        precision, recall, ndcg = evaluate_weights(
            weights,
            users_to_evaluate,
            train,
            test,
            movies,
            tfidf_matrix,
            movie_to_content_index,
            user_to_index,
            movie_ids,
            movie_to_index,
            user_factors,
            item_factors,
            popularity_stats
        )

        print(
            f"Precision@10: {precision:.4f}"
        )

        print(
            f"Recall@10:    {recall:.4f}"
        )

        print(
            f"NDCG@10:      {ndcg:.4f}"
        )

        results.append(
            {
                "content_weight": weights[0],
                "collaborative_weight": weights[1],
                "popularity_weight": weights[2],
                "precision@10": precision,
                "recall@10": recall,
                "ndcg@10": ndcg
            }
        )

    results_df = pd.DataFrame(
        results
    )

    # Best model according to NDCG
    results_df = results_df.sort_values(
        "ndcg@10",
        ascending=False
    )

    print("\n")
    print("=" * 60)
    print("HYBRID TUNING RESULTS")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    best = results_df.iloc[0]

    print("\n")
    print("=" * 60)
    print("BEST HYBRID CONFIGURATION")
    print("=" * 60)

    print(
        f"Content weight:       "
        f"{best['content_weight']:.2f}"
    )

    print(
        f"Collaborative weight: "
        f"{best['collaborative_weight']:.2f}"
    )

    print(
        f"Popularity weight:    "
        f"{best['popularity_weight']:.2f}"
    )

    print(
        f"Precision@10:         "
        f"{best['precision@10']:.4f}"
    )

    print(
        f"Recall@10:            "
        f"{best['recall@10']:.4f}"
    )

    print(
        f"NDCG@10:              "
        f"{best['ndcg@10']:.4f}"
    )

    # Save results
    results_df.to_csv(
        "ml/hybrid_tuning_results.csv",
        index=False
    )

    print(
        "\nResults saved to "
        "ml/hybrid_tuning_results.csv"
    )


if __name__ == "__main__":
    main()