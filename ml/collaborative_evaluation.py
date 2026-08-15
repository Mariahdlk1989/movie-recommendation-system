import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


MOVIES_PATH = "ml/data/movies.csv"
RATINGS_PATH = "ml/data/ratings.csv"

N_COMPONENTS = 50
TOP_K = 10

MIN_RATINGS = 20
N_USERS_TO_EVALUATE = 500

RANDOM_STATE = 42


def load_data():

    print("=" * 60)
    print("COLLABORATIVE FILTERING - MODEL EVALUATION")
    print("=" * 60)

    print("\nLoading data...")

    movies = pd.read_csv(MOVIES_PATH)

    ratings = pd.read_csv(RATINGS_PATH)

    print(f"Movies: {len(movies):,}")
    print(f"Ratings: {len(ratings):,}")

    return movies, ratings


def create_train_test_split(ratings):

    print("\nCreating train/test split...")

    rng = np.random.default_rng(RANDOM_STATE)

    train_parts = []
    test_parts = []

    grouped = ratings.groupby("userId")

    valid_users = 0

    for user_id, user_ratings in grouped:

        if len(user_ratings) < MIN_RATINGS:
            continue

        indices = np.arange(len(user_ratings))

        rng.shuffle(indices)

        split_point = int(len(indices) * 0.8)

        train_indices = indices[:split_point]
        test_indices = indices[split_point:]

        train_parts.append(
            user_ratings.iloc[train_indices]
        )

        test_parts.append(
            user_ratings.iloc[test_indices]
        )

        valid_users += 1

    train = pd.concat(
        train_parts,
        ignore_index=True
    )

    test = pd.concat(
        test_parts,
        ignore_index=True
    )

    print(
        f"Users with at least "
        f"{MIN_RATINGS} ratings: "
        f"{valid_users:,}"
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


def create_user_item_matrix(train):

    print("\nCreating User-Item matrix...")

    user_ids = train["userId"].unique()

    movie_ids = train["movieId"].unique()

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

    rows = train["userId"].map(
        user_to_index
    )

    columns = train["movieId"].map(
        movie_to_index
    )

    values = train["rating"].values

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

    return (
        matrix,
        user_ids,
        movie_ids,
        user_to_index
    )


def train_model(matrix):

    print(
        f"\nTraining SVD "
        f"with {N_COMPONENTS} components..."
    )

    svd = TruncatedSVD(
        n_components=N_COMPONENTS,
        random_state=RANDOM_STATE
    )

    user_factors = svd.fit_transform(matrix)

    explained_variance = (
        svd.explained_variance_ratio_.sum()
    )

    print(
        "Explained variance ratio:",
        explained_variance
    )

    return svd


def recommend_for_user(
    user_id,
    matrix,
    user_to_index,
    movie_ids,
    svd,
    top_k=10
):

    if user_id not in user_to_index:
        return []

    user_index = user_to_index[user_id]

    user_vector = matrix[user_index]

    latent_user = svd.transform(
        user_vector
    )

    predicted = (
        latent_user @ svd.components_
    )

    predicted = predicted.flatten()

    rated_indices = (
        user_vector.nonzero()[1]
    )

    predicted[rated_indices] = -np.inf

    top_indices = np.argsort(
        predicted
    )[::-1][:top_k]

    return [
        movie_ids[index]
        for index in top_indices
    ]


def calculate_metrics(
    recommendations,
    relevant_movies
):

    if not relevant_movies:
        return 0.0, 0.0, 0.0

    recommended_set = set(
        recommendations
    )

    relevant_set = set(
        relevant_movies
    )

    hits = len(
        recommended_set
        & relevant_set
    )

    precision = hits / len(
        recommendations
    )

    recall = hits / len(
        relevant_set
    )

    # NDCG
    dcg = 0.0

    for rank, movie_id in enumerate(
        recommendations,
        start=1
    ):

        if movie_id in relevant_set:

            dcg += 1 / np.log2(
                rank + 1
            )

    ideal_hits = min(
        len(relevant_set),
        len(recommendations)
    )

    idcg = sum(
        1 / np.log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    ndcg = (
        dcg / idcg
        if idcg > 0
        else 0.0
    )

    return (
        precision,
        recall,
        ndcg
    )


def main():

    movies, ratings = load_data()

    # -----------------------------------------
    # Train/Test split
    # -----------------------------------------

    train, test = create_train_test_split(
        ratings
    )

    # -----------------------------------------
    # User-item matrix
    # -----------------------------------------

    (
        matrix,
        user_ids,
        movie_ids,
        user_to_index
    ) = create_user_item_matrix(
        train
    )

    # -----------------------------------------
    # Train SVD
    # -----------------------------------------

    svd = train_model(matrix)

    # -----------------------------------------
    # Select evaluation users
    # -----------------------------------------

    available_users = list(
        user_to_index.keys()
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    n_users = min(
        N_USERS_TO_EVALUATE,
        len(available_users)
    )

    evaluation_users = rng.choice(
        available_users,
        size=n_users,
        replace=False
    )

    print(
        f"\nEvaluating "
        f"{n_users} users..."
    )

    precisions = []
    recalls = []
    ndcgs = []

    processed = 0

    # -----------------------------------------
    # Evaluation
    # -----------------------------------------

    for user_id in evaluation_users:

        user_test = test[
            test["userId"] == user_id
        ]

        # Relevant movies:
        # rating >= 4
        relevant_movies = (
            user_test[
                user_test["rating"] >= 4.0
            ]["movieId"]
            .tolist()
        )

        if not relevant_movies:
            continue

        recommendations = recommend_for_user(
            user_id,
            matrix,
            user_to_index,
            movie_ids,
            svd,
            TOP_K
        )

        precision, recall, ndcg = (
            calculate_metrics(
                recommendations,
                relevant_movies
            )
        )

        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)

        processed += 1

        if processed % 50 == 0:
            print(
                f"Processed "
                f"{processed}/{n_users} users..."
            )

    # -----------------------------------------
    # Results
    # -----------------------------------------

    print("\n" + "=" * 60)
    print("COLLABORATIVE FILTERING RESULTS")
    print("=" * 60)

    print(
        f"Users evaluated: "
        f"{len(precisions)}"
    )

    print(
        f"Precision@10: "
        f"{np.mean(precisions):.4f}"
    )

    print(
        f"Recall@10: "
        f"{np.mean(recalls):.4f}"
    )

    print(
        f"NDCG@10: "
        f"{np.mean(ndcgs):.4f}"
    )


if __name__ == "__main__":
    main()