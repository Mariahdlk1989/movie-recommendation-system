import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


DATA_DIR = "ml/data"
RATINGS_FILE = os.path.join(DATA_DIR, "ratings.csv")
MOVIES_FILE = os.path.join(DATA_DIR, "movies.csv")

N_COMPONENTS = 30
MIN_RATINGS = 20
TOP_N = 10


def load_data():
    print("Loading data...")

    movies = pd.read_csv(MOVIES_FILE)
    ratings = pd.read_csv(RATINGS_FILE)

    print(f"Movies: {len(movies):,}")
    print(f"Ratings: {len(ratings):,}")

    return movies, ratings


def create_train_test_split(ratings):
    print("\nCreating train/test split...")

    user_counts = ratings["userId"].value_counts()

    valid_users = user_counts[
        user_counts >= MIN_RATINGS
    ].index

    ratings = ratings[
        ratings["userId"].isin(valid_users)
    ].copy()

    # Sort by time so the last rating of every user
    # is used as the test item.
    ratings = ratings.sort_values(
        ["userId", "timestamp"]
    )

    test = ratings.groupby(
        "userId",
        group_keys=False
    ).tail(1)

    train = ratings.drop(test.index)

    print(f"Users: {len(valid_users):,}")
    print(f"Training ratings: {len(train):,}")
    print(f"Test ratings: {len(test):,}")

    return train, test


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

    print(f"Matrix shape: {matrix.shape}")
    print(f"Non-zero ratings: {matrix.nnz:,}")

    return (
        matrix,
        user_to_index,
        movie_to_index,
        user_ids,
        movie_ids
    )


def train_svd(matrix):
    print(
        f"\nTraining SVD with "
        f"{N_COMPONENTS} components..."
    )

    svd = TruncatedSVD(
        n_components=N_COMPONENTS,
        random_state=42,
        n_iter=7
    )

    user_factors = svd.fit_transform(matrix)

    item_factors = svd.components_

    explained_variance = (
        svd.explained_variance_ratio_.sum()
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

    return svd, user_factors, item_factors


def get_recommendations(
    user_id,
    movies,
    ratings,
    matrix,
    user_to_index,
    movie_to_index,
    user_ids,
    movie_ids,
    user_factors,
    item_factors,
    top_n=10
):
    if user_id not in user_to_index:
        print(f"User {user_id} not found.")
        return pd.DataFrame()

    user_index = user_to_index[user_id]

    # Predicted scores for every movie
    scores = (
        user_factors[user_index]
        @ item_factors
    )

    scores = np.asarray(scores).ravel()

    # Remove movies already rated by user
    rated_indices = matrix[
        user_index
    ].indices

    scores[rated_indices] = -np.inf

    # Get top candidates efficiently
    candidate_count = min(
        top_n * 20,
        len(scores)
    )

    candidate_indices = np.argpartition(
        -scores,
        candidate_count - 1
    )[:candidate_count]

    candidate_indices = candidate_indices[
        np.argsort(
            -scores[candidate_indices]
        )
    ]

    selected_movie_ids = [
        movie_ids[i]
        for i in candidate_indices[:top_n]
    ]

    result = movies[
        movies["movieId"].isin(
            selected_movie_ids
        )
    ].copy()

    score_map = {
        movie_ids[i]: scores[i]
        for i in candidate_indices[:top_n]
    }

    result["predicted_score"] = (
        result["movieId"].map(score_map)
    )

    result = result.sort_values(
        "predicted_score",
        ascending=False
    )

    return result[
        [
            "title",
            "genres",
            "predicted_score"
        ]
    ]


def evaluate(
    test,
    movies,
    matrix,
    user_to_index,
    movie_to_index,
    user_ids,
    movie_ids,
    user_factors,
    item_factors
):
    print("\nEvaluating model...")

    precisions = []
    recalls = []
    ndcgs = []

    evaluated = 0

    test_by_user = test.groupby("userId")

    for user_id, group in test_by_user:

        if user_id not in user_to_index:
            continue

        user_index = user_to_index[user_id]

        scores = (
            user_factors[user_index]
            @ item_factors
        )

        scores = np.asarray(scores).ravel()

        # Do not recommend training movies
        rated_indices = matrix[
            user_index
        ].indices

        scores[rated_indices] = -np.inf

        top_indices = np.argpartition(
            -scores,
            TOP_N - 1
        )[:TOP_N]

        top_indices = top_indices[
            np.argsort(
                -scores[top_indices]
            )
        ]

        recommended_movies = {
            movie_ids[i]
            for i in top_indices
        }

        relevant_movies = set(
            group["movieId"]
        )

        hits = len(
            recommended_movies
            & relevant_movies
        )

        precision = hits / TOP_N

        recall = (
            hits / len(relevant_movies)
            if relevant_movies
            else 0
        )

        # NDCG
        dcg = 0.0

        for rank, movie_id in enumerate(
            [
                movie_ids[i]
                for i in top_indices
            ],
            start=1
        ):
            if movie_id in relevant_movies:
                dcg += 1 / np.log2(rank + 1)

        ideal_hits = min(
            len(relevant_movies),
            TOP_N
        )

        idcg = sum(
            1 / np.log2(i + 1)
            for i in range(1, ideal_hits + 1)
        )

        ndcg = (
            dcg / idcg
            if idcg > 0
            else 0
        )

        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)

        evaluated += 1

        if evaluated % 50 == 0:
            print(
                f"Processed "
                f"{evaluated} users..."
            )

    return (
        evaluated,
        np.mean(precisions),
        np.mean(recalls),
        np.mean(ndcgs)
    )


def main():

    print("=" * 60)
    print("COLLABORATIVE FILTERING V2")
    print("=" * 60)

    movies, ratings = load_data()

    train, test = create_train_test_split(
        ratings
    )

    (
        matrix,
        user_to_index,
        movie_to_index,
        user_ids,
        movie_ids
    ) = create_user_item_matrix(train)

    (
        svd,
        user_factors,
        item_factors
    ) = train_svd(matrix)

    # Example recommendation
    example_user = user_ids[0]

    print(
        f"\nExample User: "
        f"{example_user}"
    )

    recommendations = get_recommendations(
        example_user,
        movies,
        ratings,
        matrix,
        user_to_index,
        movie_to_index,
        user_ids,
        movie_ids,
        user_factors,
        item_factors,
        TOP_N
    )

    print(
        "\n===== COLLABORATIVE V2 "
        "RECOMMENDATIONS ====="
    )

    print(
        recommendations.to_string(
            index=False
        )
    )

    # Evaluation
    (
        evaluated,
        precision,
        recall,
        ndcg
    ) = evaluate(
        test,
        movies,
        matrix,
        user_to_index,
        movie_to_index,
        user_ids,
        movie_ids,
        user_factors,
        item_factors
    )

    print("\n" + "=" * 60)
    print("COLLABORATIVE V2 RESULTS")
    print("=" * 60)

    print(
        f"Users evaluated: {evaluated}"
    )

    print(
        f"Precision@10: {precision:.4f}"
    )

    print(
        f"Recall@10:    {recall:.4f}"
    )

    print(
        f"NDCG@10:      {ndcg:.4f}"
    )

    print("\n" + "=" * 60)
    print("MODEL CONFIGURATION")
    print("=" * 60)

    print(
        f"SVD components: {N_COMPONENTS}"
    )

    print(
        f"Minimum ratings per user: "
        f"{MIN_RATINGS}"
    )


if __name__ == "__main__":
    main()