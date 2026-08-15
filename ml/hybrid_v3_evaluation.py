import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIG
# ============================================================

N_COMPONENTS = 30
CANDIDATE_SIZE = 200
TOP_N = 10

COLLAB_WEIGHT = 0.90
CONTENT_WEIGHT = 0.07
POPULARITY_WEIGHT = 0.03

N_USERS_EVALUATE = 500
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
# TRAIN / TEST SPLIT
# ============================================================

def create_train_test(ratings, min_ratings=20):

    print("\nCreating train/test split...")

    counts = ratings["userId"].value_counts()

    valid_users = counts[
        counts >= min_ratings
    ].index

    ratings = ratings[
        ratings["userId"].isin(valid_users)
    ].copy()

    ratings = ratings.sort_values(
        ["userId", "timestamp"]
    )

    test = ratings.groupby(
        "userId"
    ).tail(1)

    train = ratings.drop(
        test.index
    )

    print(
        f"Users: {len(valid_users):,}"
    )

    print(
        f"Training ratings: {len(train):,}"
    )

    print(
        f"Test ratings: {len(test):,}"
    )

    return train, test


# ============================================================
# USER ITEM MATRIX
# ============================================================

def create_matrix(train):

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
# SVD
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

    movies["features"] = (
        movies["title"].fillna("")
        + " "
        + movies["genres"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2)
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

    print(
        "\nCalculating popularity..."
    )

    stats = (
        train.groupby("movieId")
        .agg(
            rating_count=(
                "rating",
                "count"
            ),
            average_rating=(
                "rating",
                "mean"
            )
        )
        .reset_index()
    )

    stats["popularity_score"] = (
        np.log1p(
            stats["rating_count"]
        )
    )

    maximum = stats[
        "popularity_score"
    ].max()

    if maximum > 0:
        stats["popularity_score"] /= maximum

    return stats


# ============================================================
# NORMALIZE
# ============================================================

def normalize(scores):

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:
        return np.zeros_like(scores)

    return (
        scores - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# RECOMMEND
# ============================================================

def recommend(
    user_id,
    train,
    movies,
    tfidf,
    user_to_index,
    user_factors,
    item_factors,
    movie_ids,
    popularity
):

    if user_id not in user_to_index:
        return []

    user_index = user_to_index[
        user_id
    ]

    # --------------------------------------------------------
    # Collaborative score
    # --------------------------------------------------------

    user_vector = user_factors[
        user_index
    ]

    collaborative = (
        user_vector @ item_factors
    )

    collaborative = normalize(
        collaborative
    )

    # --------------------------------------------------------
    # Candidate movies
    # --------------------------------------------------------

    rated = set(
        train.loc[
            train["userId"] == user_id,
            "movieId"
        ]
    )

    candidate_positions = np.argsort(
        collaborative
    )[::-1]

    candidates = []

    movie_id_to_position = {
        movie_id: i
        for i, movie_id in enumerate(
            movies["movieId"]
        )
    }

    for position in candidate_positions:

        movie_id = movie_ids[position]

        if movie_id in rated:
            continue

        if movie_id not in movie_id_to_position:
            continue

        candidates.append(
            movie_id_to_position[
                movie_id
            ]
        )

        if len(candidates) >= CANDIDATE_SIZE:
            break

    if not candidates:
        return []

    candidates = np.array(
        candidates,
        dtype=int
    )

    # --------------------------------------------------------
    # Content score
    # --------------------------------------------------------

    liked = train[
        (train["userId"] == user_id)
        &
        (train["rating"] >= 4.0)
    ]

    if liked.empty:

        liked = train[
            train["userId"] == user_id
        ].nlargest(
            5,
            "rating"
        )

    liked_positions = []

    for movie_id in liked["movieId"]:

        if movie_id in movie_id_to_position:

            liked_positions.append(
                movie_id_to_position[
                    movie_id
                ]
            )

    if liked_positions:

        profile = tfidf[
            liked_positions
        ].mean(axis=0)

        profile = np.asarray(
            profile
        )

        content = cosine_similarity(
            profile,
            tfidf[candidates]
        ).ravel()

    else:

        content = np.zeros(
            len(candidates),
            dtype=np.float32
        )

    content = normalize(
        content
    )

    # --------------------------------------------------------
    # Popularity
    # --------------------------------------------------------

    popularity_map = dict(
        zip(
            popularity["movieId"],
            popularity["popularity_score"]
        )
    )

    popularity_scores = np.array(
        [
            popularity_map.get(
                movies.iloc[i]["movieId"],
                0.0
            )
            for i in candidates
        ],
        dtype=np.float32
    )

    popularity_scores = normalize(
        popularity_scores
    )

    # --------------------------------------------------------
    # Collaborative candidate scores
    # --------------------------------------------------------

    candidate_collaborative = np.array(
        [
            collaborative[
                np.where(
                    movie_ids
                    == movies.iloc[i]["movieId"]
                )[0][0]
            ]
            for i in candidates
        ],
        dtype=np.float32
    )

    # --------------------------------------------------------
    # FINAL RANKING
    # --------------------------------------------------------

    final_score = (
        COLLAB_WEIGHT
        * candidate_collaborative
        +
        CONTENT_WEIGHT
        * content
        +
        POPULARITY_WEIGHT
        * popularity_scores
    )

    order = np.argsort(
        final_score
    )[::-1]

    selected = candidates[
        order[:TOP_N]
    ]

    return [
        movies.iloc[i]["movieId"]
        for i in selected
    ]


# ============================================================
# METRICS
# ============================================================

def precision_at_k(
    recommended,
    relevant,
    k=10
):

    recommended = recommended[:k]

    if not recommended:
        return 0.0

    hits = len(
        set(recommended)
        &
        set(relevant)
    )

    return hits / k


def recall_at_k(
    recommended,
    relevant,
    k=10
):

    if not relevant:
        return 0.0

    recommended = recommended[:k]

    hits = len(
        set(recommended)
        &
        set(relevant)
    )

    return hits / len(relevant)


def ndcg_at_k(
    recommended,
    relevant,
    k=10
):

    recommended = recommended[:k]

    dcg = 0.0

    for rank, movie_id in enumerate(
        recommended,
        start=1
    ):

        if movie_id in relevant:

            dcg += 1 / np.log2(
                rank + 1
            )

    ideal_hits = min(
        len(relevant),
        k
    )

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1 / np.log2(
            rank + 1
        )
        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    return dcg / idcg


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    train,
    test,
    movies,
    tfidf,
    user_to_index,
    user_factors,
    item_factors,
    movie_ids,
    popularity
):

    print(
        f"\nEvaluating "
        f"{N_USERS_EVALUATE} users..."
    )

    users = test[
        "userId"
    ].unique()

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    if len(users) > N_USERS_EVALUATE:

        users = rng.choice(
            users,
            size=N_USERS_EVALUATE,
            replace=False
        )

    precisions = []
    recalls = []
    ndcgs = []

    processed = 0

    for user_id in users:

        relevant = test.loc[
            test["userId"] == user_id,
            "movieId"
        ].tolist()

        recommendations = recommend(
            user_id,
            train,
            movies,
            tfidf,
            user_to_index,
            user_factors,
            item_factors,
            movie_ids,
            popularity
        )

        if not recommendations:
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

        processed += 1

        if processed % 50 == 0:
            print(
                f"Processed "
                f"{processed}/"
                f"{len(users)} users..."
            )

    return (
        processed,
        np.mean(precisions),
        np.mean(recalls),
        np.mean(ndcgs)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("HYBRID V3 - MODEL EVALUATION")
    print("=" * 60)

    movies, ratings = load_data()

    train, test = create_train_test(
        ratings
    )

    (
        matrix,
        user_ids,
        movie_ids,
        user_to_index,
        movie_to_index
    ) = create_matrix(
        train
    )

    (
        user_factors,
        item_factors
    ) = train_svd(
        matrix
    )

    movies, tfidf = build_tfidf(
        movies
    )

    popularity = calculate_popularity(
        train
    )

    (
        users_evaluated,
        precision,
        recall,
        ndcg
    ) = evaluate(
        train,
        test,
        movies,
        tfidf,
        user_to_index,
        user_factors,
        item_factors,
        movie_ids,
        popularity
    )

    print("\n")
    print("=" * 60)
    print("HYBRID V3 RESULTS")
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


if __name__ == "__main__":
    main()