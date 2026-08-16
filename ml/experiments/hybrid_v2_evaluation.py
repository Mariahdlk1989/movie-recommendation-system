import os
import math
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
TOP_K = 10
EVALUATION_USERS = 500

MIN_RATINGS_PER_USER = 20

# Hybrid V2 weights
COLLAB_WEIGHT = 0.80
CONTENT_WEIGHT = 0.15
POPULARITY_WEIGHT = 0.05

# Number of collaborative candidates examined
CANDIDATE_SIZE = 200

RANDOM_STATE = 42


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

    filtered = ratings[
        ratings["userId"].isin(valid_users)
    ].copy()

    # Sort by time so the last interaction becomes test.
    filtered = filtered.sort_values(
        ["userId", "timestamp"]
    )

    test = filtered.groupby(
        "userId",
        group_keys=False
    ).tail(1)

    train = filtered.drop(
        test.index
    )

    print(f"Users: {len(valid_users):,}")
    print(f"Training ratings: {len(train):,}")
    print(f"Test ratings: {len(test):,}")

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
    ).values

    cols = train["movieId"].map(
        movie_to_index
    ).values

    values = train["rating"].values.astype(
        np.float32
    )

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
# TRAIN SVD
# ============================================================

def train_svd(matrix):
    print(
        f"\nTraining SVD with "
        f"{N_COMPONENTS} components..."
    )

    svd = TruncatedSVD(
        n_components=N_COMPONENTS,
        random_state=RANDOM_STATE
    )

    user_factors = svd.fit_transform(
        matrix
    )

    item_factors = svd.components_

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

    return (
        user_factors,
        item_factors
    )


# ============================================================
# TF-IDF
# ============================================================

def build_tfidf(movies):
    print("\nBuilding TF-IDF content model...")

    movies = movies.copy()

    movies["title"] = (
        movies["title"]
        .fillna("")
    )

    movies["genres"] = (
        movies["genres"]
        .fillna("")
        .str.replace(
            "|",
            " ",
            regex=False
        )
    )

    movies["features"] = (
        movies["title"]
        + " "
        + movies["genres"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000
    )

    tfidf = vectorizer.fit_transform(
        movies["features"]
    )

    print(
        f"TF-IDF matrix shape: "
        f"{tfidf.shape}"
    )

    return movies, tfidf


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

    max_count = stats[
        "rating_count"
    ].max()

    stats["popularity_score"] = (
        np.log1p(
            stats["rating_count"]
        )
        /
        np.log1p(max_count)
    )

    popularity_map = dict(
        zip(
            stats["movieId"],
            stats["popularity_score"]
        )
    )

    return popularity_map


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(values):

    values = np.asarray(
        values,
        dtype=np.float32
    )

    if len(values) == 0:
        return values

    minimum = values.min()
    maximum = values.max()

    if maximum - minimum < 1e-8:
        return np.zeros_like(
            values
        )

    return (
        values - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# NDCG
# ============================================================

def calculate_ndcg(recommended, relevant, k):

    recommended = recommended[:k]

    dcg = 0.0

    for rank, movie_id in enumerate(
        recommended,
        start=1
    ):
        if movie_id in relevant:
            dcg += (
                1.0
                /
                math.log2(rank + 1)
            )

    ideal_hits = min(
        len(relevant),
        k
    )

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0 / math.log2(i + 1)
        for i in range(
            1,
            ideal_hits + 1
        )
    )

    return dcg / idcg


# ============================================================
# CONTENT SCORE
# ============================================================

def get_content_scores(
    liked_movie_ids,
    candidate_movie_ids,
    movie_id_to_row,
    tfidf
):

    liked_indices = []

    for movie_id in liked_movie_ids:
        if movie_id in movie_id_to_row:
            liked_indices.append(
                movie_id_to_row[movie_id]
            )

    if not liked_indices:
        return np.zeros(
            len(candidate_movie_ids),
            dtype=np.float32
        )

    candidate_indices = []

    valid_candidates = []

    for movie_id in candidate_movie_ids:

        if movie_id in movie_id_to_row:

            candidate_indices.append(
                movie_id_to_row[movie_id]
            )

            valid_candidates.append(
                movie_id
            )

    if not candidate_indices:
        return np.zeros(
            len(candidate_movie_ids),
            dtype=np.float32
        )

    # User content profile
    profile = tfidf[
        liked_indices
    ].mean(axis=0)

    # Convert numpy matrix to ndarray.
    profile = np.asarray(
        profile
    )

    candidate_vectors = tfidf[
        candidate_indices
    ]

    # TF-IDF vectors are normalized,
    # so dot product approximates cosine similarity.
    scores = (
        candidate_vectors
        @ profile.T
    )

    scores = np.asarray(
        scores
    ).ravel()

    score_map = dict(
        zip(
            valid_candidates,
            scores
        )
    )

    return np.array([
        score_map.get(
            movie_id,
            0.0
        )
        for movie_id
        in candidate_movie_ids
    ], dtype=np.float32)


# ============================================================
# RECOMMEND FOR ONE USER
# ============================================================

def recommend_for_user(
    user_id,
    train_user,
    user_index,
    movie_ids,
    movie_to_index,
    user_factors,
    item_factors,
    movies,
    tfidf,
    movie_id_to_row,
    popularity_map
):

    # --------------------------------------------------------
    # Movies already rated
    # --------------------------------------------------------

    rated_movies = set(
        train_user["movieId"]
    )

    # Highly rated movies for content profile.
    liked_movies = train_user[
        train_user["rating"] >= 4.0
    ]["movieId"].tolist()

    # --------------------------------------------------------
    # Collaborative prediction
    # --------------------------------------------------------

    user_vector = user_factors[
        user_index
    ]

    # Shape:
    # (30,) @ (30, 82916)
    #
    # This creates only a 82k element vector,
    # NOT a huge user-item matrix.

    collab_scores = (
        user_vector
        @ item_factors
    )

    collab_scores = np.asarray(
        collab_scores,
        dtype=np.float32
    ).ravel()

    # --------------------------------------------------------
    # Remove already rated movies
    # --------------------------------------------------------

    candidate_movie_ids = []

    candidate_collab_scores = []

    for index, movie_id in enumerate(
        movie_ids
    ):

        if movie_id in rated_movies:
            continue

        candidate_movie_ids.append(
            movie_id
        )

        candidate_collab_scores.append(
            collab_scores[index]
        )

    candidate_movie_ids = np.asarray(
        candidate_movie_ids
    )

    candidate_collab_scores = np.asarray(
        candidate_collab_scores,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Candidate generation
    # --------------------------------------------------------

    candidate_collab_scores_norm = normalize(
        candidate_collab_scores
    )

    if len(candidate_movie_ids) > CANDIDATE_SIZE:

        top_indices = np.argpartition(
            candidate_collab_scores_norm,
            -CANDIDATE_SIZE
        )[-CANDIDATE_SIZE:]

        candidate_movie_ids = (
            candidate_movie_ids[
                top_indices
            ]
        )

        candidate_collab_scores_norm = (
            candidate_collab_scores_norm[
                top_indices
            ]
        )

    # --------------------------------------------------------
    # Content scores
    # --------------------------------------------------------

    content_scores = get_content_scores(
        liked_movies,
        candidate_movie_ids,
        movie_id_to_row,
        tfidf
    )

    content_scores = normalize(
        content_scores
    )

    # --------------------------------------------------------
    # Popularity
    # --------------------------------------------------------

    popularity_scores = np.array([
        popularity_map.get(
            movie_id,
            0.0
        )
        for movie_id
        in candidate_movie_ids
    ], dtype=np.float32)

    popularity_scores = normalize(
        popularity_scores
    )

    # --------------------------------------------------------
    # HYBRID SCORE
    # --------------------------------------------------------

    hybrid_scores = (
        COLLAB_WEIGHT
        * candidate_collab_scores_norm
        +
        CONTENT_WEIGHT
        * content_scores
        +
        POPULARITY_WEIGHT
        * popularity_scores
    )

    # --------------------------------------------------------
    # Top K
    # --------------------------------------------------------

    if len(hybrid_scores) > TOP_K:

        top_indices = np.argpartition(
            hybrid_scores,
            -TOP_K
        )[-TOP_K:]

        top_indices = top_indices[
            np.argsort(
                hybrid_scores[
                    top_indices
                ]
            )[::-1]
        ]

    else:
        top_indices = np.argsort(
            hybrid_scores
        )[::-1]

    return [
        candidate_movie_ids[i]
        for i in top_indices[:TOP_K]
    ]


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    train,
    test,
    movies,
    tfidf,
    user_factors,
    item_factors,
    user_to_index,
    movie_to_index,
    movie_id_to_row,
    popularity_map
):

    print(
        f"\nEvaluating "
        f"{EVALUATION_USERS} users..."
    )

    # --------------------------------------------------------
    # Group train/test by user
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

    # --------------------------------------------------------
    # Select users
    # --------------------------------------------------------

    available_users = [
        user_id
        for user_id in test_groups
        if user_id in user_to_index
        and user_id in train_groups
    ]

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    if len(available_users) > EVALUATION_USERS:

        selected_users = rng.choice(
            available_users,
            size=EVALUATION_USERS,
            replace=False
        )

    else:

        selected_users = available_users

    precision_values = []
    recall_values = []
    ndcg_values = []

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    for counter, user_id in enumerate(
        selected_users,
        start=1
    ):

        train_user = train_groups[
            user_id
        ]

        test_user = test_groups[
            user_id
        ]

        relevant_movies = set(
            test_user["movieId"]
        )

        user_index = user_to_index[
            user_id
        ]

        recommendations = recommend_for_user(
            user_id=user_id,
            train_user=train_user,
            user_index=user_index,
            movie_ids=np.array(
                list(movie_to_index.keys())
            ),
            movie_to_index=movie_to_index,
            user_factors=user_factors,
            item_factors=item_factors,
            movies=movies,
            tfidf=tfidf,
            movie_id_to_row=movie_id_to_row,
            popularity_map=popularity_map
        )

        hits = sum(
            movie_id in relevant_movies
            for movie_id in recommendations
        )

        precision = (
            hits / TOP_K
        )

        recall = (
            hits
            /
            len(relevant_movies)
            if relevant_movies
            else 0.0
        )

        ndcg = calculate_ndcg(
            recommendations,
            relevant_movies,
            TOP_K
        )

        precision_values.append(
            precision
        )

        recall_values.append(
            recall
        )

        ndcg_values.append(
            ndcg
        )

        if counter % 50 == 0:

            print(
                f"Processed "
                f"{counter}/"
                f"{len(selected_users)} users..."
            )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    precision_mean = np.mean(
        precision_values
    )

    recall_mean = np.mean(
        recall_values
    )

    ndcg_mean = np.mean(
        ndcg_values
    )

    return (
        len(selected_users),
        precision_mean,
        recall_mean,
        ndcg_mean
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("HYBRID V2 - MODEL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    movies, ratings = load_data()

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train, test = create_train_test(
        ratings
    )

    # --------------------------------------------------------
    # User-item
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
        user_factors,
        item_factors
    ) = train_svd(
        user_item_matrix
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    (
        movies,
        tfidf
    ) = build_tfidf(
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

    popularity_map = calculate_popularity(
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
        tfidf=tfidf,
        user_factors=user_factors,
        item_factors=item_factors,
        user_to_index=user_to_index,
        movie_to_index=movie_to_index,
        movie_id_to_row=movie_id_to_row,
        popularity_map=popularity_map
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("HYBRID V2 RESULTS")
    print("=" * 60)

    print(
        f"Users evaluated: "
        f"{users_evaluated}"
    )

    print(
        f"Precision@10: "
        f"{precision:.4f}"
    )

    print(
        f"Recall@10: "
        f"{recall:.4f}"
    )

    print(
        f"NDCG@10: "
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
        f"{N_COMPONENTS}"
    )

    print(
        f"Candidate size: "
        f"{CANDIDATE_SIZE}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()