import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


MOVIES_PATH = "ml/data/movies.csv"
RATINGS_PATH = "ml/data/ratings.csv"

N_COMPONENTS = 30
TOP_K = 10

MIN_RATINGS = 20
N_USERS = 500

RANDOM_STATE = 42


def load_data():
    """Load movies and ratings."""

    print("=" * 60)
    print("COLLABORATIVE FILTERING")
    print("=" * 60)

    print("\nLoading data...")

    movies = pd.read_csv(
        MOVIES_PATH
    )

    ratings = pd.read_csv(
        RATINGS_PATH
    )

    print(
        f"Movies: {len(movies):,}"
    )

    print(
        f"Ratings: {len(ratings):,}"
    )

    return movies, ratings


def create_user_item_matrix(
    ratings
):
    """
    Create sparse User-Item matrix.

    Rows    -> Users
    Columns -> Movies
    Values  -> Ratings
    """

    print(
        "\nCreating User-Item matrix..."
    )

    user_ids = ratings[
        "userId"
    ].unique()

    movie_ids = ratings[
        "movieId"
    ].unique()

    user_to_index = {
        user_id: index
        for index, user_id
        in enumerate(user_ids)
    }

    movie_to_index = {
        movie_id: index
        for index, movie_id
        in enumerate(movie_ids)
    }

    rows = ratings[
        "userId"
    ].map(
        user_to_index
    )

    columns = ratings[
        "movieId"
    ].map(
        movie_to_index
    )

    values = ratings[
        "rating"
    ].values

    matrix = csr_matrix(
        (
            values,
            (rows, columns)
        ),
        shape=(
            len(user_ids),
            len(movie_ids)
        )
    )

    print(
        "Matrix shape:",
        matrix.shape
    )

    print(
        "Non-zero ratings:",
        matrix.nnz
    )

    return (
        matrix,
        user_ids,
        movie_ids,
        user_to_index,
        movie_to_index
    )


def train_svd(
    user_item_matrix
):
    """
    Train Truncated SVD model.
    """

    print(
        f"\nTraining SVD "
        f"with {N_COMPONENTS} components..."
    )

    svd = TruncatedSVD(
        n_components=N_COMPONENTS,
        random_state=RANDOM_STATE
    )

    user_factors = svd.fit_transform(
        user_item_matrix
    )

    print(
        "User factor matrix shape:",
        user_factors.shape
    )

    print(
        "Explained variance ratio:",
        svd.explained_variance_ratio_.sum()
    )

    return svd, user_factors


def get_recommendations(
    user_id,
    user_item_matrix,
    user_ids,
    movie_ids,
    movie_to_index,
    user_to_index,
    svd,
    movies,
    top_k=10
):
    """
    Generate recommendations for a user
    using collaborative filtering.
    """

    if user_id not in user_to_index:
        return pd.DataFrame()

    user_index = user_to_index[
        user_id
    ]

    # Get user latent representation
    user_vector = user_item_matrix[
        user_index
    ]

    user_vector = svd.transform(
        user_vector
    )

    # Reconstruct predicted ratings
    predicted_ratings = (
        user_vector
        @ svd.components_
    )

    predicted_ratings = (
        predicted_ratings.flatten()
    )

    # Movies already rated
    rated_indices = (
        user_item_matrix[
            user_index
        ].nonzero()[1]
    )

    # Don't recommend already rated movies
    predicted_ratings[
        rated_indices
    ] = -np.inf

    # Top recommendations
    top_indices = np.argsort(
        predicted_ratings
    )[::-1][:top_k]

    recommended_movie_ids = [
        movie_ids[index]
        for index in top_indices
    ]

    scores = [
        predicted_ratings[index]
        for index in top_indices
    ]

    recommendations = movies[
        movies["movieId"].isin(
            recommended_movie_ids
        )
    ].copy()

    score_map = dict(
        zip(
            recommended_movie_ids,
            scores
        )
    )

    recommendations[
        "predicted_score"
    ] = recommendations[
        "movieId"
    ].map(
        score_map
    )

    recommendations = (
        recommendations
        .sort_values(
            "predicted_score",
            ascending=False
        )
    )

    return recommendations


def main():

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------

    movies, ratings = load_data()

    # --------------------------------------------------
    # 2. Create User-Item matrix
    # --------------------------------------------------

    (
        user_item_matrix,
        user_ids,
        movie_ids,
        user_to_index,
        movie_to_index
    ) = create_user_item_matrix(
        ratings
    )

    # --------------------------------------------------
    # 3. Train SVD
    # --------------------------------------------------

    (
        svd,
        user_factors
    ) = train_svd(
        user_item_matrix
    )

    # --------------------------------------------------
    # 4. Select a user
    # --------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    selected_user = rng.choice(
        user_ids
    )

    print(
        "\nExample User:",
        selected_user
    )

    # --------------------------------------------------
    # 5. Generate recommendations
    # --------------------------------------------------

    recommendations = (
        get_recommendations(
            selected_user,
            user_item_matrix,
            user_ids,
            movie_ids,
            movie_to_index,
            user_to_index,
            svd,
            movies,
            TOP_K
        )
    )

    # --------------------------------------------------
    # 6. Display results
    # --------------------------------------------------

    print(
        "\n===== COLLABORATIVE "
        "RECOMMENDATIONS ====="
    )

    if recommendations.empty:

        print(
            "No recommendations found."
        )

    else:

        print(
            recommendations[
                [
                    "title",
                    "genres",
                    "predicted_score"
                ]
            ].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()