import os
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CONFIGURATION
# ============================================================

COLLAB_WEIGHT = 0.85
CONTENT_WEIGHT = 0.10
POPULARITY_WEIGHT = 0.05

QUALITY_BONUS = 0.03

SVD_COMPONENTS = 30
TFIDF_FEATURES = 20000
CANDIDATE_POOL = 500
TOP_N = 10

EVALUATION_USERS = 500


# ============================================================
# FIND DATA FILES
# ============================================================

def find_file(filename):

    path = os.path.join(
        "ml",
        "data",
        filename
    )

    if os.path.exists(path):
        return path

    raise FileNotFoundError(
        f"Could not find {filename}: {path}"
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading data...")

    movies_path = find_file("movies.csv")
    ratings_path = find_file("ratings.csv")

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    print(f"Movies: {len(movies):,}")
    print(f"Ratings: {len(ratings):,}")

    return movies, ratings


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def train_test_split(ratings):

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
# TRAIN SVD
# ============================================================

def train_svd(user_item_matrix):

    print(
        f"\nTraining SVD with "
        f"{SVD_COMPONENTS} components..."
    )

    svd = TruncatedSVD(
        n_components=SVD_COMPONENTS,
        random_state=42
    )

    user_factors = svd.fit_transform(
        user_item_matrix
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

    return (
        user_factors,
        item_factors
    )


# ============================================================
# CONTENT MODEL
# ============================================================

def build_content_model(movies):

    print("\nBuilding TF-IDF content model...")

    movies = movies.copy()

    movies["genres"] = (
        movies["genres"]
        .fillna("")
    )

    tfidf = TfidfVectorizer(
        max_features=TFIDF_FEATURES,
        stop_words="english"
    )

    tfidf_matrix = tfidf.fit_transform(
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
        .reset_index()
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

    popularity_dict = dict(
        zip(
            popularity["movieId"],
            popularity["score"]
        )
    )

    return popularity_dict


# ============================================================
# QUALITY SCORE
# ============================================================

def calculate_quality(train):

    print("\nCalculating movie quality...")

    quality = (
        train.groupby("movieId")
        .agg(
            rating_count=("rating", "count"),
            rating_mean=("rating", "mean")
        )
        .reset_index()
    )

    # Bayesian-style quality score
    global_mean = train["rating"].mean()

    minimum_votes = 50

    quality["quality"] = (
        (
            quality["rating_count"]
            /
            (
                quality["rating_count"]
                +
                minimum_votes
            )
        )
        *
        quality["rating_mean"]
        +
        (
            minimum_votes
            /
            (
                quality["rating_count"]
                +
                minimum_votes
            )
        )
        *
        global_mean
    )

    min_quality = quality["quality"].min()
    max_quality = quality["quality"].max()

    if max_quality > min_quality:

        quality["quality_score"] = (
            quality["quality"]
            - min_quality
        ) / (
            max_quality
            - min_quality
        )

    else:

        quality["quality_score"] = 0.0

    quality_dict = dict(
        zip(
            quality["movieId"],
            quality["quality_score"]
        )
    )

    return quality_dict


# ============================================================
# CONTENT SCORES
# ============================================================

def get_content_scores(
    user_id,
    train,
    movies,
    tfidf_matrix,
    movie_content_index
):

    user_history = train[
        train["userId"] == user_id
    ]

    if len(user_history) == 0:

        return np.zeros(
            len(movies),
            dtype=np.float32
        )

    liked = user_history[
        user_history["rating"] >= 4
    ]

    if len(liked) == 0:

        liked = user_history.nlargest(
            min(5, len(user_history)),
            "rating"
        )

    liked_indices = []

    for movie_id in liked["movieId"]:

        idx = movie_content_index.get(
            movie_id
        )

        if idx is not None:
            liked_indices.append(idx)

    if not liked_indices:

        return np.zeros(
            len(movies),
            dtype=np.float32
        )

    profile = tfidf_matrix[
        liked_indices
    ].mean(axis=0)

    profile = np.asarray(
        profile
    ).ravel().astype(
        np.float32
    )

    scores = (
        tfidf_matrix @ profile
    )

    scores = np.asarray(
        scores
    ).ravel()

    min_score = scores.min()
    max_score = scores.max()

    if max_score > min_score:

        scores = (
            scores - min_score
        ) / (
            max_score - min_score
        )

    return scores.astype(
        np.float32
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

def recommend(
    user_id,
    train,
    movies,
    user_factors,
    item_factors,
    movie_ids,
    user_to_index,
    tfidf_matrix,
    movie_content_index,
    popularity_dict,
    quality_dict
):

    if user_id not in user_to_index:

        return []

    user_index = user_to_index[
        user_id
    ]

    # --------------------------------------------------------
    # Collaborative score
    # --------------------------------------------------------

    collab = (
        user_factors[user_index]
        @
        item_factors
    )

    c_min = collab.min()
    c_max = collab.max()

    if c_max > c_min:

        collab = (
            collab - c_min
        ) / (
            c_max - c_min
        )

    # --------------------------------------------------------
    # Candidate pool
    # --------------------------------------------------------

    candidate_size = min(
        CANDIDATE_POOL,
        len(collab)
    )

    candidate_indices = np.argpartition(
        collab,
        -candidate_size
    )[-candidate_size:]

    # --------------------------------------------------------
    # Content
    # --------------------------------------------------------

    content_all = get_content_scores(
        user_id,
        train,
        movies,
        tfidf_matrix,
        movie_content_index
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

        content_index = (
            movie_content_index.get(
                movie_id
            )
        )

        if content_index is None:
            content_score = 0.0

        else:

            content_score = float(
                content_all[
                    content_index
                ]
            )

        popularity_score = float(
            popularity_dict.get(
                movie_id,
                0.0
            )
        )

        quality_score = float(
            quality_dict.get(
                movie_id,
                0.0
            )
        )

        collaborative_score = float(
            collab[idx]
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
            +
            QUALITY_BONUS
            * quality_score
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
    k=10
):

    recommendations = recommendations[:k]

    if not recommendations:
        return 0.0

    hits = sum(
        movie_id in relevant
        for movie_id in recommendations
    )

    return hits / k


def recall_at_k(
    recommendations,
    relevant,
    k=10
):

    if not relevant:
        return 0.0

    recommendations = recommendations[:k]

    hits = sum(
        movie_id in relevant
        for movie_id in recommendations
    )

    return hits / len(relevant)


def ndcg_at_k(
    recommendations,
    relevant,
    k=10
):

    recommendations = recommendations[:k]

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
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 60)
    print("HYBRID V5 - MODEL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    movies, ratings = load_data()

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train, test = train_test_split(
        ratings
    )

    # --------------------------------------------------------
    # Matrix
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
    # Content
    # --------------------------------------------------------

    (
        tfidf_matrix,
        movie_content_index
    ) = build_content_model(
        movies
    )

    # --------------------------------------------------------
    # Popularity
    # --------------------------------------------------------

    popularity_dict = calculate_popularity(
        train
    )

    # --------------------------------------------------------
    # Quality
    # --------------------------------------------------------

    quality_dict = calculate_quality(
        train
    )

    # --------------------------------------------------------
    # Evaluation users
    # --------------------------------------------------------

    evaluation_users = (
        test["userId"]
        .unique()
    )

    evaluation_users = evaluation_users[
        :EVALUATION_USERS
    ]

    print(
        f"\nEvaluating "
        f"{len(evaluation_users)} users..."
    )

    precisions = []
    recalls = []
    ndcgs = []

    for i, user_id in enumerate(
        evaluation_users,
        start=1
    ):

        recommendations = recommend(
            user_id,
            train,
            movies,
            user_factors,
            item_factors,
            movie_ids,
            user_to_index,
            tfidf_matrix,
            movie_content_index,
            popularity_dict,
            quality_dict
        )

        # The held-out final rating
        user_test = test[
            test["userId"] == user_id
        ]

        relevant = set(
            user_test[
                user_test["rating"] >= 4
            ]["movieId"]
        )

        # If final rating is below 4,
        # there is no positive relevant item.
        if not relevant:
            continue

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

        if i % 50 == 0:

            print(
                f"Processed "
                f"{i}/{len(evaluation_users)} "
                f"users..."
            )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("HYBRID V5 RESULTS")
    print("=" * 60)

    if precisions:

        precision = np.mean(
            precisions
        )

        recall = np.mean(
            recalls
        )

        ndcg = np.mean(
            ndcgs
        )

        print(
            f"Users evaluated: "
            f"{len(precisions)}"
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

    else:

        print(
            "No valid evaluation users."
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
        f"Quality bonus: "
        f"{QUALITY_BONUS}"
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
        f"Users evaluated: "
        f"{len(evaluation_users)}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()