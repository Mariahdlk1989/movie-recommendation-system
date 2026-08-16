import runpy

# Run the main final model first
namespace = runpy.run_path(
    "ml/hybrid_final.py"
)

load_data = namespace["load_data"]
train_test_split = namespace["train_test_split"]
create_user_item_matrix = namespace[
    "create_user_item_matrix"
]
train_svd = namespace["train_svd"]
build_content_model = namespace[
    "build_content_model"
]
calculate_popularity = namespace[
    "calculate_popularity"
]
calculate_quality = namespace[
    "calculate_quality"
]
recommend = namespace["recommend"]

TOP_N = namespace["TOP_N"]
EVALUATION_USERS = 500


def precision_at_k(
    recommendations,
    relevant,
    k=10
):

    if not recommendations:
        return 0.0

    hits = sum(
        movie_id in relevant
        for movie_id in recommendations[:k]
    )

    return hits / k


def recall_at_k(
    recommendations,
    relevant,
    k=10
):

    if not relevant:
        return 0.0

    hits = sum(
        movie_id in relevant
        for movie_id in recommendations[:k]
    )

    return hits / len(relevant)


def ndcg_at_k(
    recommendations,
    relevant,
    k=10
):

    if not relevant:
        return 0.0

    recommendations = (
        recommendations[:k]
    )

    dcg = 0.0

    for rank, movie_id in enumerate(
        recommendations,
        start=1
    ):

        if movie_id in relevant:

            dcg += (
                1.0
                /
                __import__(
                    "numpy"
                ).log2(
                    rank + 1
                )
            )

    ideal_hits = min(
        len(relevant),
        k
    )

    if ideal_hits == 0:
        return 0.0

    np = __import__("numpy")

    idcg = sum(
        1.0
        /
        np.log2(
            rank + 1
        )
        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    return dcg / idcg


def main():

    print("=" * 60)
    print(
        "FINAL HYBRID - MODEL EVALUATION"
    )
    print("=" * 60)

    movies, ratings = (
        load_data()
    )

    train, test = (
        train_test_split(
            ratings
        )
    )

    (
        matrix,
        user_ids,
        movie_ids,
        user_to_index,
        movie_to_index
    ) = create_user_item_matrix(
        train
    )

    (
        user_factors,
        item_factors
    ) = train_svd(
        matrix
    )

    (
        tfidf_matrix,
        movie_content_index
    ) = build_content_model(
        movies
    )

    popularity_dict = (
        calculate_popularity(
            train
        )
    )

    quality_dict = (
        calculate_quality(
            train
        )
    )

    evaluation_users = (
        test["userId"]
        .unique()
        [:EVALUATION_USERS]
    )

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
            quality_dict,
            TOP_N
        )

        if hasattr(
            recommendations,
            "empty"
        ):

            if recommendations.empty:
                recommendations = []

            else:
                recommendations = (
                    recommendations[
                        "movieId"
                    ]
                    .tolist()
                )

        # IMPORTANT:
        # Keep exactly the held-out movie
        # as the relevant item only when
        # its rating is >= 4.
        user_test = test[
            test["userId"] == user_id
        ]

        relevant = set(
            user_test[
                user_test["rating"] >= 4
            ]["movieId"]
        )

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
                f"{i}/"
                f"{len(evaluation_users)} "
                f"users..."
            )

    import numpy as np

    print("\n")
    print("=" * 60)
    print(
        "FINAL HYBRID RESULTS"
    )
    print("=" * 60)

    print(
        f"Users evaluated: "
        f"{len(evaluation_users)}"
    )

    print(
        f"Precision@10: "
        f"{np.mean(precisions):.4f}"
    )

    print(
        f"Recall@10:    "
        f"{np.mean(recalls):.4f}"
    )

    print(
        f"NDCG@10:      "
        f"{np.mean(ndcgs):.4f}"
    )

    print("\n")
    print("=" * 60)
    print(
        "MODEL CONFIGURATION"
    )
    print("=" * 60)

    print(
        "Collaborative weight: 0.90"
    )

    print(
        "Content weight: 0.07"
    )

    print(
        "Popularity weight: 0.02"
    )

    print(
        "Quality weight: 0.01"
    )

    print(
        "SVD components: 30"
    )

    print(
        "Candidate pool: 500"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()