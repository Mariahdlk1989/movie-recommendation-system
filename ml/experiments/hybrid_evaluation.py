import os
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MOVIES_PATH = os.path.join(BASE_DIR, "ml", "data", "movies.csv")
RATINGS_PATH = os.path.join(BASE_DIR, "ml", "data", "ratings.csv")

N_COMPONENTS = 30
N_EVAL_USERS = 500
TOP_K = 10

# Number of candidates considered for each user
COLLAB_CANDIDATES = 300
CONTENT_CANDIDATES = 300
POPULARITY_CANDIDATES = 300

# Hybrid weights
CONTENT_WEIGHT = 0.35
COLLAB_WEIGHT = 0.50
POPULARITY_WEIGHT = 0.15

MIN_RATINGS_PER_USER = 20

RANDOM_STATE = 42


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def ndcg_at_k(recommended, relevant, k=10):
    """
    Calculate NDCG@K.
    """
    recommended = recommended[:k]

    dcg = 0.0

    for i, movie_id in enumerate(recommended):
        if movie_id in relevant:
            dcg += 1.0 / np.log2(i + 2)

    ideal_hits = min(len(relevant), k)

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0 / np.log2(i + 2)
        for i in range(ideal_hits)
    )

    return dcg / idcg


def precision_recall_at_k(recommended, relevant, k=10):
    """
    Calculate Precision@K and Recall@K.
    """

    recommended = recommended[:k]

    hits = sum(
        1 for movie_id in recommended
        if movie_id in relevant
    )

    precision = hits / k

    if len(relevant) == 0:
        recall = 0.0
    else:
        recall = hits / len(relevant)

    return precision, recall


def minmax_normalize(values):
    """
    Normalize scores to [0, 1].
    """

    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0:
        return values

    min_value = np.min(values)
    max_value = np.max(values)

    if max_value - min_value < 1e-8:
        return np.zeros_like(values)

    return (values - min_value) / (max_value - min_value)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading data...")

    movies = pd.read_csv(MOVIES_PATH)

    ratings = pd.read_csv(
        RATINGS_PATH,
        usecols=["userId", "movieId", "rating", "timestamp"]
    )

    print(f"Movies: {len(movies):,}")
    print(f"Ratings: {len(ratings):,}")

    return movies, ratings


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def create_train_test_split(ratings):

    print()
    print("Creating train/test split...")

    # Users with enough ratings
    user_counts = ratings.groupby("userId").size()

    eligible_users = user_counts[
        user_counts >= MIN_RATINGS_PER_USER
    ].index

    ratings = ratings[
        ratings["userId"].isin(eligible_users)
    ].copy()

    print(f"Users: {len(eligible_users):,}")

    # Sort chronologically
    ratings = ratings.sort_values(
        ["userId", "timestamp"]
    )

    # Last rating of each user = test rating
    test = ratings.groupby(
        "userId",
        sort=False
    ).tail(1)

    test_indices = test.index

    train = ratings.drop(
        test_indices
    )

    print(f"Training ratings: {len(train):,}")
    print(f"Test ratings: {len(test):,}")

    return train, test


# ============================================================
# BUILD USER-ITEM MATRIX
# ============================================================

def build_user_item_matrix(train):

    print()
    print("Creating User-Item matrix...")

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
        f"Non-zero ratings: {matrix.nnz:,}"
    )

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

    print()
    print(
        f"Training SVD with {N_COMPONENTS} components..."
    )

    svd = TruncatedSVD(
        n_components=N_COMPONENTS,
        random_state=RANDOM_STATE
    )

    user_factors = svd.fit_transform(
        user_item_matrix
    )

    item_factors = svd.components_.T

    print(
        "Explained variance: "
        f"{svd.explained_variance_ratio_.sum():.4f}"
    )

    print(
        f"User factor matrix: {user_factors.shape}"
    )

    print(
        f"Item factor matrix: {item_factors.shape}"
    )

    return svd, user_factors, item_factors


# ============================================================
# CONTENT MODEL
# ============================================================

def build_content_model(movies):

    print()
    print("Building TF-IDF content model...")

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

    movies["content"] = (
        movies["title"]
        + " "
        + movies["genres"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(
        movies["content"]
    )

    print(
        f"TF-IDF matrix shape: {tfidf_matrix.shape}"
    )

    movie_id_to_content_index = {
        movie_id: index
        for index, movie_id in enumerate(
            movies["movieId"]
        )
    }

    return (
        movies,
        vectorizer,
        tfidf_matrix,
        movie_id_to_content_index
    )


# ============================================================
# POPULARITY MODEL
# ============================================================

def build_popularity_model(train):

    print()
    print("Calculating popularity...")

    stats = (
        train.groupby("movieId")
        .agg(
            rating_count=("rating", "count"),
            average_rating=("rating", "mean")
        )
        .reset_index()
    )

    # Bayesian-style weighted rating
    C = stats["average_rating"].mean()

    m = stats["rating_count"].quantile(0.80)

    stats["weighted_rating"] = (
        (
            stats["rating_count"]
            /
            (
                stats["rating_count"] + m
            )
        )
        * stats["average_rating"]
        +
        (
            m
            /
            (
                stats["rating_count"] + m
            )
        )
        * C
    )

    stats["popularity_score"] = (
        stats["weighted_rating"]
        * np.log1p(stats["rating_count"])
    )

    stats["popularity_score"] = minmax_normalize(
        stats["popularity_score"].values
    )

    popularity = dict(
        zip(
            stats["movieId"],
            stats["popularity_score"]
        )
    )

    return popularity


# ============================================================
# CONTENT SCORES
# ============================================================

def get_content_candidates(
    user_movie_ids,
    tfidf_matrix,
    movie_id_to_content_index,
    movie_ids_available,
    n_candidates=300
):

    valid_indices = []

    for movie_id in user_movie_ids:

        index = movie_id_to_content_index.get(
            movie_id
        )

        if index is not None:
            valid_indices.append(index)

    if not valid_indices:
        return np.array([], dtype=np.int64)

    # User profile = average of movies rated by user
    profile = tfidf_matrix[
        valid_indices
    ].mean(axis=0)

    profile = np.asarray(
        profile
    ).reshape(1, -1)

    profile = normalize(profile)

    # Only calculate similarity against available movies
    candidate_content_indices = []

    for movie_id in movie_ids_available:

        index = movie_id_to_content_index.get(
            movie_id
        )

        if index is not None:
            candidate_content_indices.append(
                index
            )

    if not candidate_content_indices:
        return np.array([], dtype=np.int64)

    candidate_content_indices = np.asarray(
        candidate_content_indices,
        dtype=np.int64
    )

    similarities = (
        tfidf_matrix[
            candidate_content_indices
        ]
        @ profile.T
    )

    similarities = np.asarray(
        similarities
    ).ravel()

    top_count = min(
        n_candidates,
        len(similarities)
    )

    top_indices = np.argpartition(
        similarities,
        -top_count
    )[-top_count:]

    top_indices = top_indices[
        np.argsort(
            similarities[top_indices]
        )[::-1]
    ]

    return np.asarray(
        [
            movie_ids_available[i]
            for i in top_indices
        ],
        dtype=np.int64
    )


# ============================================================
# RECOMMENDATIONS FOR ONE USER
# ============================================================

def recommend_for_user(
    user_id,
    train_user_ratings,
    user_index,
    user_factors,
    item_factors,
    movie_ids,
    movie_to_index,
    tfidf_matrix,
    movie_id_to_content_index,
    popularity,
    all_movie_ids
):

    # --------------------------------------------------------
    # Movies already watched
    # --------------------------------------------------------

    watched = set(
        train_user_ratings["movieId"]
    )

    # --------------------------------------------------------
    # Collaborative scores
    # --------------------------------------------------------

    u_index = user_index.get(user_id)

    if u_index is None:
        return []

    user_vector = user_factors[
        u_index
    ]

    # IMPORTANT:
    # This calculates only ONE user's scores:
    #
    # 1 user x 82,000 movies
    #
    # instead of:
    #
    # 204,443 users x 82,000 movies
    #
    collaborative_scores = (
        item_factors @ user_vector
    )

    collaborative_scores = np.asarray(
        collaborative_scores
    ).ravel()

    # Remove watched movies
    for movie_id in watched:

        movie_index = movie_to_index.get(
            movie_id
        )

        if movie_index is not None:
            collaborative_scores[
                movie_index
            ] = -np.inf

    top_collab_count = min(
        COLLAB_CANDIDATES,
        len(collaborative_scores)
    )

    collab_indices = np.argpartition(
        collaborative_scores,
        -top_collab_count
    )[-top_collab_count:]

    collab_indices = collab_indices[
        np.argsort(
            collaborative_scores[
                collab_indices
            ]
        )[::-1]
    ]

    collab_movie_ids = [
        movie_ids[i]
        for i in collab_indices
    ]

    # --------------------------------------------------------
    # Content candidates
    # --------------------------------------------------------

    content_candidates = get_content_candidates(
        train_user_ratings["movieId"].values,
        tfidf_matrix,
        movie_id_to_content_index,
        [
            movie_id
            for movie_id in all_movie_ids
            if movie_id not in watched
        ],
        CONTENT_CANDIDATES
    )

    # --------------------------------------------------------
    # Popularity candidates
    # --------------------------------------------------------

    popularity_candidates = sorted(
        [
            movie_id
            for movie_id in popularity
            if movie_id not in watched
        ],
        key=lambda x: popularity[x],
        reverse=True
    )[:POPULARITY_CANDIDATES]

    # --------------------------------------------------------
    # Candidate pool
    # --------------------------------------------------------

    candidate_ids = set(
        collab_movie_ids
        + list(content_candidates)
        + popularity_candidates
    )

    if not candidate_ids:
        return []

    candidate_ids = list(candidate_ids)

    # --------------------------------------------------------
    # Content score for candidates
    # --------------------------------------------------------

    user_movie_ids = (
        train_user_ratings["movieId"].values
    )

    valid_profile_indices = []

    for movie_id in user_movie_ids:

        index = movie_id_to_content_index.get(
            movie_id
        )

        if index is not None:
            valid_profile_indices.append(index)

    content_score_dict = {}

    if valid_profile_indices:

        profile = tfidf_matrix[
            valid_profile_indices
        ].mean(axis=0)

        profile = np.asarray(
            profile
        ).reshape(1, -1)

        profile = normalize(profile)

        candidate_content_indices = []
        valid_candidate_ids = []

        for movie_id in candidate_ids:

            content_index = (
                movie_id_to_content_index.get(
                    movie_id
                )
            )

            if content_index is not None:

                candidate_content_indices.append(
                    content_index
                )

                valid_candidate_ids.append(
                    movie_id
                )

        if candidate_content_indices:

            similarities = (
                tfidf_matrix[
                    candidate_content_indices
                ]
                @ profile.T
            )

            similarities = np.asarray(
                similarities
            ).ravel()

            for movie_id, score in zip(
                valid_candidate_ids,
                similarities
            ):
                content_score_dict[
                    movie_id
                ] = float(score)

    # --------------------------------------------------------
    # Build final scores
    # --------------------------------------------------------

    results = []

    for movie_id in candidate_ids:

        movie_index = movie_to_index.get(
            movie_id
        )

        if movie_index is None:
            continue

        collaborative_score = (
            collaborative_scores[
                movie_index
            ]
        )

        if not np.isfinite(
            collaborative_score
        ):
            continue

        content_score = (
            content_score_dict.get(
                movie_id,
                0.0
            )
        )

        popularity_score = popularity.get(
            movie_id,
            0.0
        )

        results.append(
            (
                movie_id,
                float(collaborative_score),
                float(content_score),
                float(popularity_score)
            )
        )

    if not results:
        return []

    # --------------------------------------------------------
    # Normalize each component
    # --------------------------------------------------------

    collab_values = np.array(
        [
            x[1]
            for x in results
        ],
        dtype=np.float32
    )

    content_values = np.array(
        [
            x[2]
            for x in results
        ],
        dtype=np.float32
    )

    popularity_values = np.array(
        [
            x[3]
            for x in results
        ],
        dtype=np.float32
    )

    collab_norm = minmax_normalize(
        collab_values
    )

    content_norm = minmax_normalize(
        content_values
    )

    popularity_norm = minmax_normalize(
        popularity_values
    )

    # --------------------------------------------------------
    # Hybrid score
    # --------------------------------------------------------

    final_results = []

    for i, item in enumerate(results):

        movie_id = item[0]

        hybrid_score = (
            CONTENT_WEIGHT
            * content_norm[i]
            +
            COLLAB_WEIGHT
            * collab_norm[i]
            +
            POPULARITY_WEIGHT
            * popularity_norm[i]
        )

        final_results.append(
            (
                movie_id,
                float(hybrid_score)
            )
        )

    final_results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return final_results[:TOP_K]


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    train,
    test,
    movies,
    user_factors,
    item_factors,
    user_ids,
    movie_ids,
    user_to_index,
    movie_to_index,
    tfidf_matrix,
    movie_id_to_content_index,
    popularity
):

    print()
    print(
        f"Evaluating {N_EVAL_USERS} users..."
    )

    # Random but reproducible users
    rng = np.random.default_rng(
        RANDOM_STATE
    )

    available_users = test["userId"].unique()

    n_users = min(
        N_EVAL_USERS,
        len(available_users)
    )

    evaluation_users = rng.choice(
        available_users,
        size=n_users,
        replace=False
    )

    # --------------------------------------------------------
    # Group training ratings by user
    # --------------------------------------------------------

    train_groups = {
        user_id: group
        for user_id, group
        in train.groupby("userId")
    }

    test_groups = {
        user_id: group
        for user_id, group
        in test.groupby("userId")
    }

    movie_id_set = set(
        movies["movieId"].values
    )

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    processed = 0

    for user_id in evaluation_users:

        train_user = train_groups.get(
            user_id
        )

        test_user = test_groups.get(
            user_id
        )

        if train_user is None:
            continue

        if test_user is None:
            continue

        relevant = set(
            test_user["movieId"]
        )

        recommendations = recommend_for_user(
            user_id=user_id,
            train_user_ratings=train_user,
            user_index=user_to_index,
            user_factors=user_factors,
            item_factors=item_factors,
            movie_ids=movie_ids,
            movie_to_index=movie_to_index,
            tfidf_matrix=tfidf_matrix,
            movie_id_to_content_index=movie_id_to_content_index,
            popularity=popularity,
            all_movie_ids=list(movie_id_set)
        )

        recommended_ids = [
            movie_id
            for movie_id, score
            in recommendations
        ]

        if not recommended_ids:
            continue

        precision, recall = (
            precision_recall_at_k(
                recommended_ids,
                relevant,
                TOP_K
            )
        )

        ndcg = ndcg_at_k(
            recommended_ids,
            relevant,
            TOP_K
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

        processed += 1

        if processed % 50 == 0:
            print(
                f"Processed {processed}/{n_users} users..."
            )

    return (
        processed,
        np.mean(precision_scores)
        if precision_scores else 0.0,
        np.mean(recall_scores)
        if recall_scores else 0.0,
        np.mean(ndcg_scores)
        if ndcg_scores else 0.0
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("HYBRID MODEL - FAST EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    movies, ratings = load_data()

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    train, test = create_train_test_split(
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
    ) = build_user_item_matrix(
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
        tfidf_matrix,
        movie_id_to_content_index
    ) = build_content_model(
        movies
    )

    # --------------------------------------------------------
    # Popularity
    # --------------------------------------------------------

    popularity = build_popularity_model(
        train
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    (
        users_evaluated,
        precision,
        recall,
        ndcg
    ) = evaluate(
        train=train,
        test=test,
        movies=movies,
        user_factors=user_factors,
        item_factors=item_factors,
        user_ids=user_ids,
        movie_ids=movie_ids,
        user_to_index=user_to_index,
        movie_to_index=movie_to_index,
        tfidf_matrix=tfidf_matrix,
        movie_id_to_content_index=movie_id_to_content_index,
        popularity=popularity
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("HYBRID MODEL RESULTS")
    print("=" * 60)

    print(
        f"Users evaluated: {users_evaluated}"
    )

    print(
        f"Precision@10: {precision:.4f}"
    )

    print(
        f"Recall@10: {recall:.4f}"
    )

    print(
        f"NDCG@10: {ndcg:.4f}"
    )

    print()
    print("=" * 60)
    print("MODEL CONFIGURATION")
    print("=" * 60)

    print(
        f"Content weight: {CONTENT_WEIGHT}"
    )

    print(
        f"Collaborative weight: {COLLAB_WEIGHT}"
    )

    print(
        f"Popularity weight: {POPULARITY_WEIGHT}"
    )

    print(
        f"SVD components: {N_COMPONENTS}"
    )

    print(
        f"Users evaluated: {users_evaluated}"
    )


if __name__ == "__main__":
    main()