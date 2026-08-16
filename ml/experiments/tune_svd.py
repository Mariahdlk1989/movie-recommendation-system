import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


MOVIES_PATH = "ml/data/movies.csv"
RATINGS_PATH = "ml/data/ratings.csv"

COMPONENTS_TO_TEST = [
    20,
    30,
    50,
    75,
    100
]

MIN_RATINGS = 20
N_USERS_TO_EVALUATE = 300
TOP_K = 10

RANDOM_STATE = 42


def load_ratings():

    print("Loading ratings...")

    ratings = pd.read_csv(
        RATINGS_PATH
    )

    print(
        f"Ratings: {len(ratings):,}"
    )

    return ratings


def create_train_test_split(
    ratings
):

    print(
        "\nCreating train/test split..."
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    train_parts = []
    test_parts = []

    valid_users = []

    for user_id, user_ratings in ratings.groupby(
        "userId"
    ):

        if len(user_ratings) < MIN_RATINGS:
            continue

        indices = np.arange(
            len(user_ratings)
        )

        rng.shuffle(indices)

        split = int(
            len(indices) * 0.8
        )

        train_parts.append(
            user_ratings.iloc[
                indices[:split]
            ]
        )

        test_parts.append(
            user_ratings.iloc[
                indices[split:]
            ]
        )

        valid_users.append(
            user_id
        )

    train = pd.concat(
        train_parts,
        ignore_index=True
    )

    test = pd.concat(
        test_parts,
        ignore_index=True
    )

    print(
        f"Users: {len(valid_users):,}"
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


def create_matrix(
    train
):

    print(
        "\nCreating sparse matrix..."
    )

    user_ids = train[
        "userId"
    ].unique()

    movie_ids = train[
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

    rows = train[
        "userId"
    ].map(
        user_to_index
    )

    columns = train[
        "movieId"
    ].map(
        movie_to_index
    )

    values = train[
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

    return (
        matrix,
        user_ids,
        movie_ids,
        user_to_index
    )


def recommend(
    user_id,
    matrix,
    user_to_index,
    movie_ids,
    svd
):

    user_index = user_to_index[
        user_id
    ]

    user_vector = matrix[
        user_index
    ]

    latent_user = svd.transform(
        user_vector
    )

    predictions = (
        latent_user
        @ svd.components_
    )

    predictions = (
        predictions.flatten()
    )

    rated_movies = (
        user_vector.nonzero()[1]
    )

    predictions[
        rated_movies
    ] = -np.inf

    top_indices = np.argsort(
        predictions
    )[::-1][:TOP_K]

    return [
        movie_ids[index]
        for index in top_indices
    ]


def metrics(
    recommendations,
    relevant
):

    if not relevant:
        return 0, 0, 0

    relevant = set(
        relevant
    )

    hits = [
        movie
        for movie in recommendations
        if movie in relevant
    ]

    precision = (
        len(hits)
        / TOP_K
    )

    recall = (
        len(hits)
        / len(relevant)
    )

    dcg = 0

    for rank, movie in enumerate(
        recommendations,
        start=1
    ):

        if movie in relevant:

            dcg += (
                1
                / np.log2(rank + 1)
            )

    ideal_hits = min(
        len(relevant),
        TOP_K
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
        else 0
    )

    return (
        precision,
        recall,
        ndcg
    )


def evaluate_model(
    components,
    matrix,
    user_ids,
    movie_ids,
    user_to_index,
    test
):

    print(
        f"\nTraining SVD "
        f"with {components} components..."
    )

    svd = TruncatedSVD(
        n_components=components,
        random_state=RANDOM_STATE
    )

    svd.fit(
        matrix
    )

    explained_variance = (
        svd.explained_variance_ratio_.sum()
    )

    print(
        f"Explained variance: "
        f"{explained_variance:.4f}"
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    users = list(
        user_to_index.keys()
    )

    n_users = min(
        N_USERS_TO_EVALUATE,
        len(users)
    )

    selected_users = rng.choice(
        users,
        size=n_users,
        replace=False
    )

    precisions = []
    recalls = []
    ndcgs = []

    for user_id in selected_users:

        user_test = test[
            test["userId"] == user_id
        ]

        relevant = (
            user_test[
                user_test["rating"] >= 4
            ]["movieId"]
            .tolist()
        )

        if not relevant:
            continue

        recommendations = recommend(
            user_id,
            matrix,
            user_to_index,
            movie_ids,
            svd
        )

        p, r, n = metrics(
            recommendations,
            relevant
        )

        precisions.append(p)
        recalls.append(r)
        ndcgs.append(n)

    return (
        np.mean(precisions),
        np.mean(recalls),
        np.mean(ndcgs),
        explained_variance
    )


def main():

    print("=" * 60)
    print("SVD HYPERPARAMETER TUNING")
    print("=" * 60)

    ratings = load_ratings()

    train, test = (
        create_train_test_split(
            ratings
        )
    )

    (
        matrix,
        user_ids,
        movie_ids,
        user_to_index
    ) = create_matrix(
        train
    )

    results = []

    for components in COMPONENTS_TO_TEST:

        (
            precision,
            recall,
            ndcg,
            variance
        ) = evaluate_model(
            components,
            matrix,
            user_ids,
            movie_ids,
            user_to_index,
            test
        )

        results.append({
            "components": components,
            "precision@10": precision,
            "recall@10": recall,
            "ndcg@10": ndcg,
            "explained_variance": variance
        })

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "SVD TUNING RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    best = results_df.loc[
        results_df[
            "ndcg@10"
        ].idxmax()
    ]

    print(
        "\nBEST MODEL"
    )

    print(
        f"Components: "
        f"{int(best['components'])}"
    )

    print(
        f"Precision@10: "
        f"{best['precision@10']:.4f}"
    )

    print(
        f"Recall@10: "
        f"{best['recall@10']:.4f}"
    )

    print(
        f"NDCG@10: "
        f"{best['ndcg@10']:.4f}"
    )


if __name__ == "__main__":
    main()