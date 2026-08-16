
import os
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# FINAL HYBRID MODEL EVALUATION
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

COLLAB_WEIGHT = 0.90
CONTENT_WEIGHT = 0.08
POPULARITY_WEIGHT = 0.02

SVD_COMPONENTS = 30
TFIDF_FEATURES = 20000

CANDIDATE_POOL = 300

EVALUATION_USERS = 500

RELEVANT_THRESHOLD = 4.0

K_VALUES = [5, 10, 20]

RANDOM_STATE = 42

DATA_DIR = os.path.join(
    "ml",
    "data"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading data...")

    movies_path = os.path.join(
        DATA_DIR,
        "movies.csv"
    )

    ratings_path = os.path.join(
        DATA_DIR,
        "ratings.csv"
    )

    if not os.path.exists(movies_path):
        raise FileNotFoundError(
            f"Could not find: {movies_path}"
        )

    if not os.path.exists(ratings_path):
        raise FileNotFoundError(
            f"Could not find: {ratings_path}"
        )

    movies = pd.read_csv(
        movies_path
    )

    ratings = pd.read_csv(
        ratings_path
    )

    print(
        f"Movies: {len(movies):,}"
    )

    print(
        f"Ratings: {len(ratings):,}"
    )

    return movies, ratings


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def create_split(ratings):

    print(
        "\nCreating train/test split..."
    )

    # Sort by user and time.
    ratings = ratings.sort_values(
        ["userId", "timestamp"]
    )

    # Last interaction of every user
    # becomes the test interaction.
    test = ratings.groupby(
        "userId",
        sort=False
    ).tail(1)

    # Remaining interactions are training.
    train = ratings.drop(
        test.index
    )

    print(
        f"Users: "
        f"{ratings['userId'].nunique():,}"
    )

    print(
        f"Training ratings: "
        f"{len(train):,}"
    )

    print(
        f"Test ratings: "
        f"{len(test):,}"
    )

    return train, test


# ============================================================
# USER-ITEM MATRIX
# ============================================================

def create_user_item_matrix(train):

    print(
        "\nCreating User-Item matrix..."
    )

    user_ids = train[
        "userId"
    ].unique()

    movie_ids = train[
        "movieId"
    ].unique()

    user_to_index = {
        user_id: index
        for index, user_id in enumerate(
            user_ids
        )
    }

    movie_to_index = {
        movie_id: index
        for index, movie_id in enumerate(
            movie_ids
        )
    }

    rows = train[
        "userId"
    ].map(
        user_to_index
    ).to_numpy()

    cols = train[
        "movieId"
    ].map(
        movie_to_index
    ).to_numpy()

    values = train[
        "rating"
    ].astype(
        np.float32
    ).to_numpy()

    matrix = csr_matrix(
        (
            values,
            (rows, cols)
        ),
        shape=(
            len(user_ids),
            len(movie_ids)
        ),
        dtype=np.float32
    )

    print(
        f"Matrix shape: "
        f"{matrix.shape}"
    )

    print(
        f"Non-zero ratings: "
        f"{matrix.nnz:,}"
    )

    return (
        matrix,
        user_ids,
        movie_ids,
        user_to_index,
        movie_to_index
    )


# ============================================================
# SVD
# ============================================================

def train_svd(matrix):

    print(
        f"\nTraining SVD with "
        f"{SVD_COMPONENTS} components..."
    )

    svd = TruncatedSVD(
        n_components=SVD_COMPONENTS,
        random_state=RANDOM_STATE
    )

    user_factors = svd.fit_transform(
        matrix
    ).astype(
        np.float32
    )

    item_factors = (
        svd.components_
        .astype(np.float32)
    )

    explained_variance = (
        svd.explained_variance_ratio_
        .sum()
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

    return (
        user_factors,
        item_factors
    )


# ============================================================
# TF-IDF CONTENT MODEL
# ============================================================

def build_tfidf(movies):

    print(
        "\nBuilding TF-IDF content model..."
    )

    movies = movies.copy()

    movies["genres"] = (
        movies["genres"]
        .fillna("")
        .str.replace(
            "|",
            " ",
            regex=False
        )
    )

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_FEATURES
    )

    tfidf_matrix = vectorizer.fit_transform(
        movies["genres"]
    )

    print(
        f"TF-IDF matrix shape: "
        f"{tfidf_matrix.shape}"
    )

    movie_content_index = {
        movie_id: index
        for index, movie_id in enumerate(
            movies["movieId"]
        )
    }

    return (
        tfidf_matrix,
        movie_content_index
    )


# ============================================================
# POPULARITY
# ============================================================

def calculate_popularity(train):

    print(
        "\nCalculating popularity..."
    )

    popularity = (
        train
        .groupby("movieId")
        .agg(
            rating_count=(
                "rating",
                "count"
            ),
            rating_mean=(
                "rating",
                "mean"
            )
        )
    )

    popularity["score"] = (
        np.log1p(
            popularity["rating_count"]
        )
        *
        popularity["rating_mean"]
    )

    max_score = popularity[
        "score"
    ].max()

    if max_score > 0:

        popularity["score"] /= (
            max_score
        )

    return popularity[
        "score"
    ].to_dict()


# ============================================================
# NORMALIZE
# ============================================================

def normalize_scores(scores):

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    minimum = scores.min()
    maximum = scores.max()

    if maximum > minimum:

        scores = (
            scores - minimum
        ) / (
            maximum - minimum
        )

    else:

        scores = np.zeros_like(
            scores
        )

    return scores


# ============================================================
# USER CONTENT PROFILE
# ============================================================

def build_user_content_profile(
    user_id,
    train,
    tfidf_matrix,
    movie_content_index
):

    history = train[
        train["userId"] == user_id
    ]

    if history.empty:

        return np.zeros(
            tfidf_matrix.shape[1],
            dtype=np.float32
        )

    # Ratings >= 4 are considered
    # positive preferences.
    liked = history[
        history["rating"]
        >= RELEVANT_THRESHOLD
    ]

    # Fallback for users without
    # any positive rating.
    if liked.empty:

        liked = history.nlargest(
            min(5, len(history)),
            "rating"
        )

    indices = []

    for movie_id in liked[
        "movieId"
    ]:

        index = movie_content_index.get(
            movie_id
        )

        if index is not None:

            indices.append(
                index
            )

    if not indices:

        return np.zeros(
            tfidf_matrix.shape[1],
            dtype=np.float32
        )

    profile = tfidf_matrix[
        indices
    ].mean(
        axis=0
    )

    return np.asarray(
        profile
    ).ravel().astype(
        np.float32
    )


# ============================================================
# HYBRID RECOMMENDATIONS
# ============================================================

def generate_hybrid_recommendations(
    user_id,
    train,
    user_factors,
    item_factors,
    movie_ids,
    user_to_index,
    tfidf_matrix,
    movie_content_index,
    popularity_dict,
    top_n
):

    if user_id not in user_to_index:

        return []

    user_index = user_to_index[
        user_id
    ]

    # --------------------------------------------------------
    # Collaborative filtering
    # --------------------------------------------------------

    user_vector = user_factors[
        user_index
    ]

    collaborative = (
        user_vector
        @ item_factors
    )

    collaborative = normalize_scores(
        collaborative
    )

    # --------------------------------------------------------
    # Candidate pool
    # --------------------------------------------------------

    candidate_size = min(
        CANDIDATE_POOL,
        len(collaborative)
    )

    candidate_indices = np.argpartition(
        collaborative,
        -candidate_size
    )[-candidate_size:]

    # --------------------------------------------------------
    # Content-based filtering
    # --------------------------------------------------------

    profile = build_user_content_profile(
        user_id,
        train,
        tfidf_matrix,
        movie_content_index
    )

    content_scores = (
        tfidf_matrix @ profile
    )

    content_scores = np.asarray(
        content_scores
    ).ravel()

    content_scores = normalize_scores(
        content_scores
    )

    # --------------------------------------------------------
    # Watched movies
    # --------------------------------------------------------

    watched = set(
        train[
            train["userId"] == user_id
        ]["movieId"]
    )

    recommendations = []

    # --------------------------------------------------------
    # Calculate hybrid score
    # --------------------------------------------------------

    for index in candidate_indices:

        movie_id = movie_ids[
            index
        ]

        # Do not recommend watched movies.
        if movie_id in watched:

            continue

        content_index = (
            movie_content_index.get(
                movie_id
            )
        )

        if content_index is None:

            content_score = 0.0

        else:

            content_score = float(
                content_scores[
                    content_index
                ]
            )

        popularity_score = float(
            popularity_dict.get(
                movie_id,
                0.0
            )
        )

        collaborative_score = float(
            collaborative[index]
        )

        hybrid_score = (
            COLLAB_WEIGHT
            * collaborative_score
            +
            CONTENT_WEIGHT
            * content_score
            +
            POPULARITY_WEIGHT
            * popularity_score
        )

        recommendations.append(
            (
                movie_id,
                hybrid_score
            )
        )

    recommendations.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return [
        movie_id
        for movie_id, score
        in recommendations[:top_n]
    ]


# ============================================================
# POPULARITY BASELINE
# ============================================================

def generate_popularity_recommendations(
    user_id,
    train,
    popularity_dict,
    top_n
):

    # Movies already watched.
    watched = set(
        train[
            train["userId"] == user_id
        ]["movieId"]
    )

    # Sort every movie by popularity.
    ranked_movies = sorted(
        popularity_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for movie_id, score in ranked_movies:

        if movie_id in watched:

            continue

        recommendations.append(
            movie_id
        )

        if len(recommendations) >= top_n:

            break

    return recommendations


# ============================================================
# PRECISION@K
# ============================================================

def precision_at_k(
    recommendations,
    relevant,
    k
):

    recommendations = (
        recommendations[:k]
    )

    if not recommendations:

        return 0.0

    hits = sum(
        movie_id in relevant
        for movie_id in recommendations
    )

    return hits / k


# ============================================================
# RECALL@K
# ============================================================

def recall_at_k(
    recommendations,
    relevant,
    k
):

    if not relevant:

        return 0.0

    recommendations = (
        recommendations[:k]
    )

    hits = sum(
        movie_id in relevant
        for movie_id in recommendations
    )

    return hits / len(relevant)


# ============================================================
# NDCG@K
# ============================================================

def ndcg_at_k(
    recommendations,
    relevant,
    k
):

    if not relevant:

        return 0.0

    recommendations = (
        recommendations[:k]
    )

    dcg = 0.0

    for rank, movie_id in enumerate(
        recommendations,
        start=1
    ):

        if movie_id in relevant:

            dcg += (
                1.0
                /
                np.log2(rank + 1)
            )

    ideal_hits = min(
        len(relevant),
        k
    )

    if ideal_hits == 0:

        return 0.0

    idcg = sum(
        1.0
        /
        np.log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    return dcg / idcg


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model_name,
    users,
    test,
    train,
    recommendation_function
):

    print("\n")
    print("=" * 60)
    print(
        f"EVALUATING {model_name}"
    )
    print("=" * 60)

    results = {
        k: {
            "precision": [],
            "recall": [],
            "ndcg": []
        }
        for k in K_VALUES
    }

    for counter, user_id in enumerate(
        users,
        start=1
    ):

        user_test = test[
            test["userId"] == user_id
        ]

        # Only test ratings >= 4
        # are considered relevant.
        relevant = set(
            user_test[
                user_test["rating"]
                >= RELEVANT_THRESHOLD
            ]["movieId"]
        )

        if not relevant:

            continue

        recommendations = (
            recommendation_function(
                user_id
            )
        )

        for k in K_VALUES:

            results[k][
                "precision"
            ].append(
                precision_at_k(
                    recommendations,
                    relevant,
                    k
                )
            )

            results[k][
                "recall"
            ].append(
                recall_at_k(
                    recommendations,
                    relevant,
                    k
                )
            )

            results[k][
                "ndcg"
            ].append(
                ndcg_at_k(
                    recommendations,
                    relevant,
                    k
                )
            )

        if counter % 50 == 0:

            print(
                f"Processed "
                f"{counter}/"
                f"{len(users)} users..."
            )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "FINAL HYBRID MODEL EVALUATION"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    movies, ratings = load_data()

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train, test = create_split(
        ratings
    )

    # --------------------------------------------------------
    # User-item matrix
    # --------------------------------------------------------

    (
        matrix,
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
        user_factors,
        item_factors
    ) = train_svd(
        matrix
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    (
        tfidf_matrix,
        movie_content_index
    ) = build_tfidf(
        movies
    )

    # --------------------------------------------------------
    # Popularity
    # --------------------------------------------------------

    popularity_dict = (
        calculate_popularity(
            train
        )
    )

    # --------------------------------------------------------
    # Select evaluation users
    # --------------------------------------------------------

    eligible_users = [
        user_id
        for user_id in test["userId"]
        if user_id in user_to_index
    ]

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    if len(eligible_users) > EVALUATION_USERS:

        evaluation_users = rng.choice(
            eligible_users,
            size=EVALUATION_USERS,
            replace=False
        )

    else:

        evaluation_users = np.array(
            eligible_users
        )

    print(
        f"\nUsers selected for evaluation: "
        f"{len(evaluation_users)}"
    )

    # --------------------------------------------------------
    # Hybrid function
    # --------------------------------------------------------

    def hybrid_function(user_id):

        return generate_hybrid_recommendations(
            user_id,
            train,
            user_factors,
            item_factors,
            movie_ids,
            user_to_index,
            tfidf_matrix,
            movie_content_index,
            popularity_dict,
            max(K_VALUES)
        )

    # --------------------------------------------------------
    # Popularity function
    # --------------------------------------------------------

    def popularity_function(user_id):

        return generate_popularity_recommendations(
            user_id,
            train,
            popularity_dict,
            max(K_VALUES)
        )

    # --------------------------------------------------------
    # Evaluate Hybrid
    # --------------------------------------------------------

    hybrid_results = evaluate_model(
        "FINAL HYBRID MODEL",
        evaluation_users,
        test,
        train,
        hybrid_function
    )

    # --------------------------------------------------------
    # Evaluate Popularity baseline
    # --------------------------------------------------------

    popularity_results = evaluate_model(
        "POPULARITY BASELINE",
        evaluation_users,
        test,
        train,
        popularity_function
    )

    # --------------------------------------------------------
    # Final table
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print(
        "FINAL EVALUATION RESULTS"
    )
    print("=" * 80)

    print(
        f"{'Model':<24}"
        f"{'K':<6}"
        f"{'Precision':<15}"
        f"{'Recall':<15}"
        f"{'NDCG':<15}"
    )

    print("-" * 80)

    for k in K_VALUES:

        precision = np.mean(
            hybrid_results[k]["precision"]
        )

        recall = np.mean(
            hybrid_results[k]["recall"]
        )

        ndcg = np.mean(
            hybrid_results[k]["ndcg"]
        )

        print(
            f"{'Hybrid Final':<24}"
            f"{k:<6}"
            f"{precision:<15.4f}"
            f"{recall:<15.4f}"
            f"{ndcg:<15.4f}"
        )

    print("-" * 80)

    for k in K_VALUES:

        precision = np.mean(
            popularity_results[k][
                "precision"
            ]
        )

        recall = np.mean(
            popularity_results[k][
                "recall"
            ]
        )

        ndcg = np.mean(
            popularity_results[k][
                "ndcg"
            ]
        )

        print(
            f"{'Popularity Baseline':<24}"
            f"{k:<6}"
            f"{precision:<15.4f}"
            f"{recall:<15.4f}"
            f"{ndcg:<15.4f}"
        )

    print("=" * 80)

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print(
        "FINAL MODEL CONFIGURATION"
    )
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
        f"{SVD_COMPONENTS}"
    )

    print(
        f"TF-IDF features: "
        f"{TFIDF_FEATURES}"
    )

    print(
        f"Candidate pool: "
        f"{CANDIDATE_POOL}"
    )

    print(
        f"Evaluation users: "
        f"{len(evaluation_users)}"
    )

    print(
        f"Relevant threshold: "
        f"rating >= {RELEVANT_THRESHOLD:g}"
    )

    print(
        f"K values: "
        f"{K_VALUES}"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
