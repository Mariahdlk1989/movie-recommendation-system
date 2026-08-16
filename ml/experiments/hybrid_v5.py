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

SVD_COMPONENTS = 30
TFIDF_FEATURES = 20000

CANDIDATE_POOL = 500
TOP_N = 10

# Minimum number of ratings used for confidence
MIN_RATINGS_FOR_QUALITY = 20

DATA_DIR = "data"


# ============================================================
# FIND DATA FILES
# ============================================================

def find_file(filename):

    path = os.path.join("ml", "data", filename)

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
# MOVIE QUALITY
# ============================================================

def calculate_movie_quality(train):

    print("\nCalculating movie quality...")

    stats = (
        train.groupby("movieId")
        .agg(
            rating_count=("rating", "count"),
            rating_mean=("rating", "mean")
        )
        .reset_index()
    )

    # Confidence grows with the number of ratings,
    # but saturates so extremely popular movies
    # do not dominate.

    confidence = (
        stats["rating_count"]
        /
        (
            stats["rating_count"]
            + MIN_RATINGS_FOR_QUALITY
        )
    )

    # Rating quality centered around the dataset scale.
    rating_quality = (
        stats["rating_mean"] / 5.0
    )

    stats["quality_score"] = (
        confidence * rating_quality
    )

    max_quality = stats["quality_score"].max()

    if max_quality > 0:
        stats["quality_score"] /= max_quality

    quality_dict = dict(
        zip(
            stats["movieId"],
            stats["quality_score"]
        )
    )

    return quality_dict


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
# COLLABORATIVE SCORE
# ============================================================

def collaborative_scores(
    user_index,
    user_factors,
    item_factors
):

    user_vector = user_factors[
        user_index
    ]

    scores = user_vector @ item_factors

    return scores


# ============================================================
# CONTENT SCORE
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
    quality_dict,
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

    user_index = user_to_index[
        user_id
    ]

    # --------------------------------------------------------
    # Collaborative
    # --------------------------------------------------------

    collab = collaborative_scores(
        user_index,
        user_factors,
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
    # Existing movies
    # --------------------------------------------------------

    watched = set(
        train[
            train["userId"] == user_id
        ]["movieId"]
    )

    rows = []

    for idx in candidate_indices:

        movie_id = movie_ids[idx]

        if movie_id in watched:
            continue

        movie_row = movies[
            movies["movieId"] == movie_id
        ]

        if movie_row.empty:
            continue

        movie_index = movie_content_index.get(
            movie_id
        )

        if movie_index is None:

            content_score = 0.0

        else:

            content_score = float(
                content_all[movie_index]
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

        # ----------------------------------------------------
        # V5 Ranking
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

        # Quality is used as a small ranking refinement.
        #
        # It does NOT replace collaborative filtering.
        # It only helps distinguish between similarly
        # ranked candidates.

        quality_bonus = (
            0.03 * quality_score
        )

        final_score = (
            hybrid_score
            +
            quality_bonus
        )

        row = movie_row.iloc[0]

        rows.append({

            "movieId":
                movie_id,

            "title":
                row["title"],

            "genres":
                row["genres"],

            "collaborative_score":
                collaborative_score,

            "content_score":
                content_score,

            "popularity_score":
                popularity_score,

            "quality_score":
                quality_score,

            "hybrid_score":
                final_score
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
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "HYBRID MOVIE RECOMMENDATION SYSTEM V5"
    )
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
    # Quality
    # --------------------------------------------------------

    quality_dict = calculate_movie_quality(
        train
    )

    # --------------------------------------------------------
    # Example
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
        popularity_dict,
        quality_dict
    )

    print("\n")
    print("=" * 60)
    print("HYBRID V5 RECOMMENDATIONS")
    print("=" * 60)

    if not recommendations.empty:

        display_columns = [

            "title",

            "genres",

            "collaborative_score",

            "content_score",

            "popularity_score",

            "quality_score",

            "hybrid_score"
        ]

        print(
            recommendations[
                display_columns
            ].to_string(index=False)
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

    print(
        "Quality bonus: 0.03"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()