
import os
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# FINAL HYBRID MOVIE RECOMMENDATION SYSTEM
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
TOP_N = 10

RANDOM_STATE = 42

DATA_DIR = os.path.join(
    "ml",
    "data"
)


# ============================================================
# FIND DATA FILES
# ============================================================

def find_file(filename):

    path = os.path.join(
        DATA_DIR,
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

    movies_path = find_file(
        "movies.csv"
    )

    ratings_path = find_file(
        "ratings.csv"
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

def train_test_split(ratings):

    print(
        "\nCreating train/test split..."
    )

    # Sort chronologically for every user.
    ratings = ratings.sort_values(
        ["userId", "timestamp"]
    )

    # Last rating of each user is used
    # as the test interaction.
    test = ratings.groupby(
        "userId",
        sort=False
    ).tail(1)

    # Everything else is training data.
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

    # Users and movies appearing in training data.
    user_ids = train[
        "userId"
    ].unique()

    movie_ids = train[
        "movieId"
    ].unique()

    # Map original IDs to matrix indices.
    user_to_index = {
        user_id: i
        for i, user_id in enumerate(
            user_ids
        )
    }

    movie_to_index = {
        movie_id: i
        for i, movie_id in enumerate(
            movie_ids
        )
    }

    # Convert IDs into matrix positions.
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

    # Sparse user-item matrix.
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
        random_state=RANDOM_STATE
    )

    # User latent representation.
    user_factors = svd.fit_transform(
        user_item_matrix
    ).astype(
        np.float32
    )

    # Item latent representation.
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

    print(
        "\nBuilding TF-IDF content model..."
    )

    movies = movies.copy()

    # Replace missing genres.
    movies["genres"] = (
        movies["genres"]
        .fillna("")
    )

    # Convert:
    # Action|Adventure|Sci-Fi
    #
    # into:
    # Action Adventure Sci-Fi
    movies["genres"] = (
        movies["genres"]
        .str.replace(
            "|",
            " ",
            regex=False
        )
    )

    tfidf = TfidfVectorizer(
        max_features=TFIDF_FEATURES
    )

    tfidf_matrix = tfidf.fit_transform(
        movies["genres"]
    )

    print(
        f"TF-IDF matrix shape: "
        f"{tfidf_matrix.shape}"
    )

    # Map movieId to row in TF-IDF matrix.
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

    # Combine number of ratings and
    # average rating.
    popularity["score"] = (
        np.log1p(
            popularity["rating_count"]
        )
        *
        popularity["rating_mean"]
    )

    # Normalize to [0, 1].
    max_score = popularity[
        "score"
    ].max()

    if max_score > 0:

        popularity["score"] /= (
            max_score
        )

    popularity_dict = (
        popularity["score"]
        .to_dict()
    )

    return popularity_dict


# ============================================================
# NORMALIZE SCORES
# ============================================================

def normalize_scores(scores):

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    min_score = scores.min()
    max_score = scores.max()

    if max_score > min_score:

        scores = (
            scores - min_score
        ) / (
            max_score - min_score
        )

    else:

        scores = np.zeros_like(
            scores
        )

    return scores


# ============================================================
# COLLABORATIVE SCORES
# ============================================================

def collaborative_scores(
    user_index,
    user_factors,
    item_factors
):

    user_vector = user_factors[
        user_index
    ]

    # Result is a vector with one
    # score for every movie.
    scores = (
        user_vector @ item_factors
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

    # Movies rated >= 4 are considered liked.
    liked = history[
        history["rating"] >= 4
    ]

    # If the user has no rating >= 4,
    # use their highest-rated movies.
    if liked.empty:

        liked = history.nlargest(
            min(5, len(history)),
            "rating"
        )

    liked_indices = []

    for movie_id in liked[
        "movieId"
    ]:

        index = movie_content_index.get(
            movie_id
        )

        if index is not None:

            liked_indices.append(
                index
            )

    if not liked_indices:

        return np.zeros(
            tfidf_matrix.shape[1],
            dtype=np.float32
        )

    # Average TF-IDF representation
    # of movies liked by the user.
    profile = tfidf_matrix[
        liked_indices
    ].mean(
        axis=0
    )

    profile = np.asarray(
        profile
    ).ravel().astype(
        np.float32
    )

    return profile


# ============================================================
# CONTENT SCORES
# ============================================================

def get_content_scores(
    user_id,
    train,
    tfidf_matrix,
    movie_content_index
):

    profile = build_user_content_profile(
        user_id,
        train,
        tfidf_matrix,
        movie_content_index
    )

    # Compare user profile against
    # every movie's TF-IDF representation.
    scores = (
        tfidf_matrix @ profile
    )

    scores = np.asarray(
        scores
    ).ravel()

    # Normalize to [0, 1].
    scores = normalize_scores(
        scores
    )

    return scores


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
    top_n=TOP_N
):

    # --------------------------------------------------------
    # Unknown user
    # --------------------------------------------------------

    if user_id not in user_to_index:

        print(
            f"User {user_id} not found. "
            f"Using popular movies."
        )

        popular = movies.copy()

        popular[
            "popularity_score"
        ] = (
            popular["movieId"]
            .map(popularity_dict)
            .fillna(0)
        )

        return popular.nlargest(
            top_n,
            "popularity_score"
        )


    # --------------------------------------------------------
    # User index
    # --------------------------------------------------------

    user_index = user_to_index[
        user_id
    ]


    # --------------------------------------------------------
    # Collaborative filtering
    # --------------------------------------------------------

    collaborative = collaborative_scores(
        user_index,
        user_factors,
        item_factors
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
    # Content-based scores
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
        train[
            train["userId"] == user_id
        ]["movieId"]
    )


    # --------------------------------------------------------
    # Calculate hybrid scores
    # --------------------------------------------------------

    rows = []

    for index in candidate_indices:

        movie_id = movie_ids[
            index
        ]

        # Never recommend movies
        # already watched by the user.
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


        # Popularity score.
        popularity_score = float(
            popularity_dict.get(
                movie_id,
                0.0
            )
        )


        # Collaborative score.
        collaborative_score = float(
            collaborative[index]
        )


        # Final weighted hybrid score.
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


        # Find movie information.
        movie_rows = movies[
            movies["movieId"] == movie_id
        ]

        if movie_rows.empty:

            continue

        movie = movie_rows.iloc[0]


        rows.append({

            "movieId":
                movie_id,

            "title":
                movie["title"],

            "genres":
                movie["genres"],

            "collaborative_score":
                collaborative_score,

            "content_score":
                content_score,

            "popularity_score":
                popularity_score,

            "hybrid_score":
                hybrid_score
        })


    # --------------------------------------------------------
    # No recommendations
    # --------------------------------------------------------

    if not rows:

        return pd.DataFrame()


    # --------------------------------------------------------
    # Sort recommendations
    # --------------------------------------------------------

    result = pd.DataFrame(
        rows
    )

    result = result.sort_values(
        "hybrid_score",
        ascending=False
    )

    return result.head(
        top_n
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "FINAL HYBRID MOVIE "
        "RECOMMENDATION SYSTEM"
    )
    print("=" * 60)


    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    movies, ratings = load_data()


    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------

    train, test = train_test_split(
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
    # Content model
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

    popularity_dict = (
        calculate_popularity(
            train
        )
    )


    # --------------------------------------------------------
    # Example user
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


    # --------------------------------------------------------
    # Generate recommendations
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Display recommendations
    # --------------------------------------------------------

    print("\n")

    print("=" * 60)
    print("FINAL RECOMMENDATIONS")
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
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "No recommendations "
            "could be generated."
        )


    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print("\n")

    print("=" * 60)
    print("FINAL MODEL CONFIGURATION")
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
        f"Top N: "
        f"{TOP_N}"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
