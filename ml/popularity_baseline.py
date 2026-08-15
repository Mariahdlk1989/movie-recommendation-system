import pandas as pd
import numpy as np


MOVIES_PATH = "ml/data/movies.csv"
RATINGS_PATH = "ml/data/ratings.csv"

TOP_K = 10
MIN_RATINGS = 20
N_USERS = 500

RANDOM_STATE = 42


def load_data():
    """Load movies and ratings."""

    print("Loading data...")

    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)

    print(f"Movies: {len(movies):,}")
    print(f"Ratings: {len(ratings):,}")

    return movies, ratings


def calculate_movie_popularity(
    movies,
    ratings
):
    """
    Calculate movie popularity using
    rating count and average rating.
    """

    movie_stats = (
        ratings
        .groupby("movieId")
        .agg(
            rating_count=("rating", "count"),
            average_rating=("rating", "mean")
        )
        .reset_index()
    )

    # Merge movie information
    movie_stats = movie_stats.merge(
        movies[
            ["movieId", "title", "genres"]
        ],
        on="movieId",
        how="left"
    )

    # Sort by number of ratings
    movie_stats = movie_stats.sort_values(
        "rating_count",
        ascending=False
    )

    return movie_stats


def get_popular_movies(
    movie_stats,
    top_k
):
    """Return the most popular movies."""

    popular_movies = (
        movie_stats
        .head(top_k)
        ["movieId"]
        .tolist()
    )

    return popular_movies


def get_valid_users(ratings):
    """
    Find users with at least
    MIN_RATINGS ratings.
    """

    user_counts = (
        ratings
        .groupby("userId")
        .size()
    )

    valid_users = user_counts[
        user_counts >= MIN_RATINGS
    ].index

    return valid_users


def train_test_split(
    ratings,
    user_id
):
    """
    Split user ratings chronologically.

    First 80% -> training
    Last 20%  -> testing
    """

    user_ratings = ratings[
        ratings["userId"] == user_id
    ].copy()

    user_ratings = user_ratings.sort_values(
        "timestamp"
    )

    split_index = int(
        len(user_ratings) * 0.8
    )

    train = user_ratings.iloc[
        :split_index
    ]

    test = user_ratings.iloc[
        split_index:
    ]

    return train, test


def get_recommendations(
    train,
    movie_stats,
    k
):
    """
    Recommend globally popular movies
    that the user has not already rated.
    """

    rated_movies = set(
        train["movieId"]
    )

    recommendations = []

    for movie_id in movie_stats[
        "movieId"
    ]:

        if movie_id not in rated_movies:

            recommendations.append(
                movie_id
            )

        if len(recommendations) >= k:
            break

    return recommendations


def precision_at_k(
    recommended,
    relevant,
    k
):
    """Calculate Precision@K."""

    recommended = recommended[:k]

    if len(recommended) == 0:
        return 0.0

    relevant = set(relevant)

    hits = len(
        set(recommended) & relevant
    )

    return hits / len(recommended)


def recall_at_k(
    recommended,
    relevant,
    k
):
    """Calculate Recall@K."""

    relevant = set(relevant)

    if len(relevant) == 0:
        return 0.0

    recommended = recommended[:k]

    hits = len(
        set(recommended) & relevant
    )

    return hits / len(relevant)


def dcg_at_k(
    relevance_scores,
    k
):
    """Calculate DCG@K."""

    relevance_scores = (
        relevance_scores[:k]
    )

    if len(relevance_scores) == 0:
        return 0.0

    dcg = 0.0

    for i, relevance in enumerate(
        relevance_scores
    ):

        position = i + 1

        dcg += (
            relevance
            / np.log2(position + 1)
        )

    return dcg


def ndcg_at_k(
    recommended,
    relevant,
    k
):
    """Calculate NDCG@K."""

    recommended = recommended[:k]

    relevant = set(relevant)

    if len(recommended) == 0:
        return 0.0

    relevance_scores = [
        1 if movie_id in relevant
        else 0
        for movie_id in recommended
    ]

    ideal_scores = sorted(
        relevance_scores,
        reverse=True
    )

    dcg = dcg_at_k(
        relevance_scores,
        k
    )

    ideal_dcg = dcg_at_k(
        ideal_scores,
        k
    )

    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg


def main():

    print("=" * 60)
    print(
        "MOVIE RECOMMENDATION "
        "- POPULARITY BASELINE"
    )
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------

    movies, ratings = load_data()

    # --------------------------------------------------
    # 2. Calculate movie popularity
    # --------------------------------------------------

    print(
        "\nCalculating movie popularity..."
    )

    movie_stats = calculate_movie_popularity(
        movies,
        ratings
    )

    print("\nTop 10 most popular movies:")

    print(
        movie_stats[
            [
                "title",
                "rating_count",
                "average_rating"
            ]
        ].head(10).to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # 3. Find valid users
    # --------------------------------------------------

    valid_users = get_valid_users(
        ratings
    )

    print(
        f"\nUsers with at least "
        f"{MIN_RATINGS} ratings: "
        f"{len(valid_users):,}"
    )

    # --------------------------------------------------
    # 4. Select users
    # --------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    selected_users = rng.choice(
        valid_users,
        size=min(
            N_USERS,
            len(valid_users)
        ),
        replace=False
    )

    print(
        f"Evaluating "
        f"{len(selected_users)} users..."
    )

    # --------------------------------------------------
    # 5. Evaluation arrays
    # --------------------------------------------------

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    evaluated_users = 0

    # --------------------------------------------------
    # 6. Evaluate users
    # --------------------------------------------------

    for i, user_id in enumerate(
        selected_users,
        start=1
    ):

        train, test = train_test_split(
            ratings,
            user_id
        )

        # Movies the user liked
        # in the test set
        relevant_movies = test[
            test["rating"] >= 4.0
        ]["movieId"].tolist()

        if len(relevant_movies) == 0:
            continue

        recommendations = (
            get_recommendations(
                train,
                movie_stats,
                TOP_K
            )
        )

        if not recommendations:
            continue

        precision = precision_at_k(
            recommendations,
            relevant_movies,
            TOP_K
        )

        recall = recall_at_k(
            recommendations,
            relevant_movies,
            TOP_K
        )

        ndcg = ndcg_at_k(
            recommendations,
            relevant_movies,
            TOP_K
        )

        precision_scores.append(
            precision
        )

        recall_scores.append(
            recall
        )

        ndcg_scores.append(
            ndcg
        )

        evaluated_users += 1

        if i % 50 == 0:

            print(
                f"Processed {i}/"
                f"{len(selected_users)} users..."
            )

    # --------------------------------------------------
    # 7. Results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("POPULARITY BASELINE RESULTS")
    print("=" * 60)

    if evaluated_users == 0:

        print(
            "No users could be evaluated."
        )

        return

    mean_precision = np.mean(
        precision_scores
    )

    mean_recall = np.mean(
        recall_scores
    )

    mean_ndcg = np.mean(
        ndcg_scores
    )

    print(
        f"Users evaluated: "
        f"{evaluated_users}"
    )

    print(
        f"Precision@{TOP_K}: "
        f"{mean_precision:.4f}"
    )

    print(
        f"Recall@{TOP_K}: "
        f"{mean_recall:.4f}"
    )

    print(
        f"NDCG@{TOP_K}: "
        f"{mean_ndcg:.4f}"
    )


if __name__ == "__main__":
    main()