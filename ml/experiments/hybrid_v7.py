import os
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CONFIGURATION
# ============================================================

COLLAB_WEIGHT = 0.90
CONTENT_WEIGHT = 0.08
POPULARITY_WEIGHT = 0.02

SVD_COMPONENTS = 30
TFIDF_FEATURES = 20000

CANDIDATE_POOL = 500
TOP_N = 10

EVALUATION_USERS = 500

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading data...")

    movies_path = os.path.join(
        "ml",
        "data",
        "movies.csv"
    )

    ratings_path = os.path.join(
        "ml",
        "data",
        "ratings.csv"
    )

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    print(f"Movies: {len(movies):,}")
    print(f"Ratings: {len(ratings):,}")

    return movies, ratings


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def create_split(ratings):

    print("\nCreating train/test split...")

    ratings = ratings.sort_values(
        ["userId", "timestamp"]
    )

    test = ratings.groupby(
        "userId",
        sort=False
    ).tail(1)

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
# USER ITEM MATRIX
# ============================================================

def create_user_item_matrix(train):

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

    rows = train["userId"].map(
        user_to_index
    ).to_numpy()

    cols = train["movieId"].map(
        movie_to_index
    ).to_numpy()

    values = train["rating"].astype(
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
        f"Matrix shape: {matrix.shape}"
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
    ).astype(np.float32)

    item_factors = svd.components_.astype(
        np.float32
    )

    explained = (
        svd.explained_variance_ratio_.sum()
    )

    print(
        f"Explained variance: "
        f"{explained:.4f}"
    )

    print(
        f"User factor matrix: "
        f"{user_factors.shape}"
    )

    print(
        f"Item factor matrix: "
        f"{item_factors.shape}"
    )

    return user_factors, item_factors


# ============================================================
# TF-IDF
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
        movie_id: i
        for i, movie_id in enumerate(
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

    print("\nCalculating popularity...")

    popularity = (
        train.groupby("movieId")
        .agg(
            rating_count=("rating", "count"),
            rating_mean=("rating", "mean")
        )
    )

    popularity["score"] = (
        np.log1p(
            popularity["rating_count"]
        )
        *
        popularity["rating_mean"]
    )

    max_score = popularity["score"].max()

    if max_score > 0:

        popularity["score"] /= max_score

    return popularity["score"].to_dict()


# ============================================================
# NORMALIZE
# ============================================================

def normalize_scores(scores):

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    min_value = scores.min()
    max_value = scores.max()

    if max_value > min_value:

        scores = (
            scores - min_value
        ) / (
            max_value - min_value
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

    # Give stronger influence to highly-rated movies.
    liked = history[
        history["rating"] >= 4
    ].copy()

    if liked.empty:

        liked = history.nlargest(
            min(5, len(history)),
            "rating"
        ).copy()

    indices = []
    weights = []

    for _, row in liked.iterrows():

        movie_id = row["movieId"]

        idx = movie_content_index.get(
            movie_id
        )

        if idx is not None:

            indices.append(idx)

            # Rating 4 -> 1.0
            # Rating 5 -> 2.0
            weight = max(
                0.1,
                float(row["rating"]) - 3.0
            )

            weights.append(weight)

    if not indices:

        return np.zeros(
            tfidf_matrix.shape[1],
            dtype=np.float32
        )

    selected = tfidf_matrix[
        indices
    ]

    weights = np.asarray(
        weights,
        dtype=np.float32
    )

    profile = selected.multiply(
        weights.reshape(-1, 1)
    ).sum(axis=0)

    profile = np.asarray(
        profile
    ).ravel()

    weight_sum = weights.sum()

    if weight_sum > 0:

        profile /= weight_sum

    return profile.astype(
        np.float32
    )


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    user_id,
    train,
    movies,
    user_factors,
    item_factors,
    movie_ids,
    user_to_index,
    tfidf_matrix,
    movie_content_index,
    popularity_dict
):

    if user_id not in user_to_index:

        return []

    user_index = user_to_index[
        user_id
    ]

    # --------------------------------------------------------
    # Collaborative
    # --------------------------------------------------------

    user_vector = user_factors[
        user_index
    ]

    collaborative = (
        user_vector @ item_factors
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
    # Content profile
    # --------------------------------------------------------

    profile = build_user_content_profile(
        user_id,
        train,
        tfidf_matrix,
        movie_content_index
    )

    # --------------------------------------------------------
    # Only calculate content scores for
    # candidates instead of every movie
    # --------------------------------------------------------

    content_indices = []

    valid_candidates = []

    for idx in candidate_indices:

        movie_id = movie_ids[idx]

        content_idx = movie_content_index.get(
            movie_id
        )

        if content_idx is not None:

            content_indices.append(
                content_idx
            )

            valid_candidates.append(
                idx
            )

    candidate_content_scores = {}

    if content_indices:

        candidate_matrix = tfidf_matrix[
            content_indices
        ]

        content_scores = (
            candidate_matrix @ profile
        )

        content_scores = np.asarray(
            content_scores
        ).ravel()

        content_scores = normalize_scores(
            content_scores
        )

        for idx, score in zip(
            valid_candidates,
            content_scores
        ):

            candidate_content_scores[idx] = float(
                score
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

    for idx in candidate_indices:

        movie_id = movie_ids[idx]

        if movie_id in watched:

            continue

        collaborative_score = float(
            collaborative[idx]
        )

        content_score = candidate_content_scores.get(
            idx,
            0.0
        )

        popularity_score = float(
            popularity_dict.get(
                movie_id,
                0.0
            )
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
        in recommendations[:TOP_N]
    ]


# ============================================================
# METRICS
# ============================================================

def precision_at_k(
    recommendations,
    relevant,
    k
):

    recommendations = recommendations[:k]

    if not recommendations:

        return 0.0

    hits = sum(
        movie in relevant
        for movie in recommendations
    )

    return hits / k


def recall_at_k(
    recommendations,
    relevant,
    k
):

    if not relevant:

        return 0.0

    recommendations = recommendations[:k]

    hits = sum(
        movie in relevant
        for movie in recommendations
    )

    return hits / len(relevant)


def ndcg_at_k(
    recommendations,
    relevant,
    k
):

    recommendations = recommendations[:k]

    dcg = 0.0

    for i, movie_id in enumerate(
        recommendations
    ):

        if movie_id in relevant:

            dcg += 1.0 / np.log2(
                i + 2
            )

    ideal_hits = min(
        len(relevant),
        k
    )

    if ideal_hits == 0:

        return 0.0

    idcg = sum(
        1.0 / np.log2(i + 2)
        for i in range(
            ideal_hits
        )
    )

    return dcg / idcg


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("HYBRID V7 - MODEL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    movies, ratings = load_data()

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train, test = create_split(
        ratings
    )

    # --------------------------------------------------------
    # Matrix
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

    popularity_dict = calculate_popularity(
        train
    )

    # --------------------------------------------------------
    # Evaluation users
    # --------------------------------------------------------

    test_users = [
        user_id
        for user_id in test["userId"]
        if user_id in user_to_index
    ]

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    if len(test_users) > EVALUATION_USERS:

        test_users = rng.choice(
            test_users,
            size=EVALUATION_USERS,
            replace=False
        )

    print(
        f"\nEvaluating "
        f"{len(test_users)} users..."
    )

    precisions = []
    recalls = []
    ndcgs = []

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    for counter, user_id in enumerate(
        test_users,
        start=1
    ):

        user_test = test[
            test["userId"] == user_id
        ]

        relevant = set(
            user_test[
                user_test["rating"] >= 4
            ]["movieId"]
        )

        if not relevant:

            continue

        recommendations = (
            generate_recommendations(
                user_id,
                train,
                movies,
                user_factors,
                item_factors,
                movie_ids,
                user_to_index,
                tfidf_matrix,
                movie_content_index,
                popularity_dict
            )
        )

        precisions.append(
            precision_at_k(
                recommendations,
                relevant,
                TOP_N
            )
        )

        recalls.append(
            recall_at_k(
                recommendations,
                relevant,
                TOP_N
            )
        )

        ndcgs.append(
            ndcg_at_k(
                recommendations,
                relevant,
                TOP_N
            )
        )

        if counter % 50 == 0:

            print(
                f"Processed "
                f"{counter}/"
                f"{len(test_users)} users..."
            )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    precision = np.mean(
        precisions
    )

    recall = np.mean(
        recalls
    )

    ndcg = np.mean(
        ndcgs
    )

    print("\n")
    print("=" * 60)
    print("HYBRID V7 RESULTS")
    print("=" * 60)

    print(
        f"Users evaluated: "
        f"{len(test_users)}"
    )

    print(
        f"Precision@10: "
        f"{precision:.4f}"
    )

    print(
        f"Recall@10:    "
        f"{recall:.4f}"
    )

    print(
        f"NDCG@10:      "
        f"{ndcg:.4f}"
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
        f"{SVD_COMPONENTS}"
    )

    print(
        f"Candidate size: "
        f"{CANDIDATE_POOL}"
    )

    print(
        "Weighted content profile: True"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()