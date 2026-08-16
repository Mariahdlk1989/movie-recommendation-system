
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

# A rating is considered relevant for evaluation
RELEVANT_RATING = 4.0

DATA_DIR = os.path.join("ml", "data")


# ============================================================
# FIND DATA FILES
# ============================================================

def find_file(filename):

    path = os.path.join(DATA_DIR, filename)

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
        svd,
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
        .replace("(no genres listed)", "")
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

        popularity["score"] = (
            popularity["score"] / max_score
        )

    return dict(
        zip(
            popularity["movieId"],
            popularity["score"]
        )
    )


# ============================================================
# COLLABORATIVE SCORE
# ============================================================

def collaborative_scores(
    user_index,
    user_factors,
    item_factors
):

    user_vector = user_factors[user_index]

    scores = user_vector @ item_factors

    return scores.astype(np.float32)


# ============================================================
# CONTENT SCORE
# ============================================================

def get_content_scores(
    user_id,
    train,
    tfidf_matrix,
    movie_content_index
):

    user_history = train[
        train["userId"] == user_id
    ]

    if len(user_history) == 0:

        return np.zeros(
            tfidf_matrix.shape[0],
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
            tfidf_matrix.shape[0],
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

    scores = tfidf_matrix @ profile

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

    return scores.astype(np.float32)


# ============================================================
# NORMALIZE
# ============================================================

def minmax_normalize(scores):

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    min_value = scores.min()
    max_value = scores.max()

    if max_value > min_value:

        return (
            (scores - min_value)
            /
            (max_value - min_value)
        ).astype(np.float32)

    return np.zeros_like(
        scores,
        dtype=np.float32
    )


# ============================================================
# RECOMMEND
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
    top_n=TOP_N
):

    if user_id not in user_to_index:

        print(
            f"User {user_id} not found. "
            f"Using popular movies."
        )

        popular = movies.copy()

        popular["popularity_score"] = (
            popular["movieId"]
            .map(popularity_dict)
            .fillna(0)
        )

        return popular.nlargest(
            top_n,
            "popularity_score"
        )

    user_index = user_to_index[user_id]

    # --------------------------------------------------------
    # Collaborative
    # --------------------------------------------------------

    collab = collaborative_scores(
        user_index,
        user_factors,
        item_factors
    )

    collab = minmax_normalize(
        collab
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
        tfidf_matrix,
        movie_content_index
    )

    # --------------------------------------------------------
    # Watched movies
    # --------------------------------------------------------

    watched = set(
        train.loc[
            train["userId"] == user_id,
            "movieId"
        ]
    )

    # --------------------------------------------------------
    # Build recommendations
    # --------------------------------------------------------

    rows = []

    for idx in candidate_indices:

        movie_id = movie_ids[idx]

        # Never recommend already watched movies
        if movie_id in watched:
            continue

        movie_index = movie_content_index.get(
            movie_id
        )

        if movie_index is None:
            continue

        movie_rows = movies[
            movies["movieId"] == movie_id
        ]

        if movie_rows.empty:
            continue

        content_score = float(
            content_all[movie_index]
        )

        popularity_score = float(
            popularity_dict.get(
                movie_id,
                0.0
            )
        )

        collaborative_score = float(
            collab[idx]
        )

        # ----------------------------------------------------
        # Hybrid score
        # ----------------------------------------------------

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

        row = movie_rows.iloc[0]

        rows.append({
            "movieId": movie_id,
            "title": row["title"],
            "genres": row["genres"],
            "collaborative_score":
                collaborative_score,
            "content_score":
                content_score,
            "popularity_score":
                popularity_score,
            "hybrid_score":
                hybrid_score
        })

    if not rows:

        return pd.DataFrame()

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "hybrid_score",
        ascending=False
    )

    return result.head(top_n)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    train,
    test,
    movies,
    user_factors,
    item_factors,
    movie_ids,
    user_to_index,
    tfidf_matrix,
    movie_content_index,
    popularity_dict
):

    print("\nEvaluating 500 users...")

    # Same evaluation philosophy as V4
    evaluation_users = test[
        test["userId"].isin(
            user_to_index.keys()
        )
    ]

    evaluation_users = evaluation_users.head(
        500
    )

    precision_values = []
    recall_values = []
    ndcg_values = []

    processed = 0

    for _, test_row in evaluation_users.iterrows():

        user_id = test_row["userId"]
        actual_movie = test_row["movieId"]
        actual_rating = test_row["rating"]

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
            TOP_N
        )

        if recommendations.empty:
            continue

        recommended_movies = (
            recommendations["movieId"]
            .tolist()
        )

        hit = (
            actual_movie in recommended_movies
            and actual_rating >= RELEVANT_RATING
        )

        precision = (
            1.0 / TOP_N
            if hit
            else 0.0
        )

        recall = (
            1.0
            if hit
            else 0.0
        )

        if hit:

            rank = (
                recommended_movies.index(
                    actual_movie
                )
                + 1
            )

            ndcg = (
                1.0
                /
                np.log2(rank + 1)
            )

        else:

            ndcg = 0.0

        precision_values.append(
            precision
        )

        recall_values.append(
            recall
        )

        ndcg_values.append(
            ndcg
        )

        processed += 1

        if processed % 50 == 0:

            print(
                f"Processed "
                f"{processed}/500 users..."
            )

    if not precision_values:

        return 0.0, 0.0, 0.0, 0

    precision = np.mean(
        precision_values
    )

    recall = np.mean(
        recall_values
    )

    ndcg = np.mean(
        ndcg_values
    )

    return (
        precision,
        recall,
        ndcg,
        processed
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("HYBRID MOVIE RECOMMENDATION SYSTEM V6")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    movies, ratings = load_data()

    # --------------------------------------------------------
    # Train / Test
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
        svd,
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
    # Example recommendation
    # --------------------------------------------------------

    example_user = int(
        user_ids[0]
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
        example_user,
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

    print("\n")
    print("=" * 60)
    print("HYBRID V6 RECOMMENDATIONS")
    print("=" * 60)

    if not recommendations.empty:

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
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    (
        precision,
        recall,
        ndcg,
        evaluated
    ) = evaluate_model(
        train,
        test,
        movies,
        user_factors,
        item_factors,
        movie_ids,
        user_to_index,
        tfidf_matrix,
        movie_content_index,
        popularity_dict
    )

    print("\n")
    print("=" * 60)
    print("FINAL HYBRID V6 RESULTS")
    print("=" * 60)

    print(
        f"Users evaluated: "
        f"{evaluated}"
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
        f"Candidate pool: "
        f"{CANDIDATE_POOL}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()