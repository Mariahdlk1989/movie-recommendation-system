import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MOVIES_PATH = "ml/data/movies.csv"
RATINGS_PATH = "ml/data/ratings.csv"

TOP_K = 10
MIN_RATINGS = 20
N_USERS = 500

RANDOM_STATE = 42


def load_data():
    print("Loading data...")

    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)

    print(f"Movies: {len(movies):,}")
    print(f"Ratings: {len(ratings):,}")

    return movies, ratings


def prepare_movies(movies):

    movies = movies.copy()

    movies["genres"] = (
        movies["genres"]
        .fillna("")
        .str.replace("|", " ", regex=False)
    )

    movies["features"] = (
        movies["title"].fillna("")
        + " "
        + movies["genres"]
    )

    return movies


def build_content_model(movies):

    print("\nBuilding TF-IDF model...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000
    )

    tfidf_matrix = vectorizer.fit_transform(
        movies["features"]
    )

    print(
        "TF-IDF matrix shape:",
        tfidf_matrix.shape
    )

    return vectorizer, tfidf_matrix


def get_valid_users(ratings):

    user_counts = ratings.groupby("userId").size()

    valid_users = user_counts[
        user_counts >= MIN_RATINGS
    ].index

    return valid_users


def train_test_split(ratings, user_id):

    user_ratings = ratings[
        ratings["userId"] == user_id
    ].copy()

    # Sort chronologically
    user_ratings = user_ratings.sort_values(
        "timestamp"
    )

    # Last 20% for testing
    split_index = int(
        len(user_ratings) * 0.8
    )

    train = user_ratings.iloc[:split_index]
    test = user_ratings.iloc[split_index:]

    return train, test


def get_recommendations(
    train,
    movies,
    tfidf_matrix
):

    # Movies the user has already rated
    rated_movies = set(
        train["movieId"]
    )

    # Use movies rated highly by the user
    liked_movies = train[
        train["rating"] >= 4.0
    ]

    if liked_movies.empty:
        return []

    movie_id_to_index = pd.Series(
        movies.index,
        index=movies["movieId"]
    )

    liked_indices = []

    for movie_id in liked_movies["movieId"]:

        if movie_id in movie_id_to_index:
            liked_indices.append(
                movie_id_to_index[movie_id]
            )

    if not liked_indices:
        return []

    # Calculate similarity against liked movies
    similarity_matrix = cosine_similarity(
        tfidf_matrix[liked_indices],
        tfidf_matrix
    )

    # Average similarity across liked movies
    scores = similarity_matrix.mean(axis=0)

    # Don't recommend movies already rated
    for movie_id in rated_movies:

        if movie_id in movie_id_to_index:

            index = movie_id_to_index[movie_id]

            scores[index] = -1

    # Top K recommendations
    top_indices = np.argsort(
        scores
    )[::-1][:TOP_K]

    recommendations = movies.iloc[
        top_indices
    ]["movieId"].tolist()

    return recommendations


def precision_at_k(
    recommended,
    relevant,
    k
):

    recommended = recommended[:k]

    if len(recommended) == 0:
        return 0.0

    hits = len(
        set(recommended)
        & set(relevant)
    )

    return hits / len(recommended)


def recall_at_k(
    recommended,
    relevant,
    k
):

    relevant = set(relevant)

    if len(relevant) == 0:
        return 0.0

    recommended = recommended[:k]

    hits = len(
        set(recommended)
        & relevant
    )

    return hits / len(relevant)


def main():

    print("=" * 60)
    print("MOVIE RECOMMENDATION - MODEL EVALUATION")
    print("=" * 60)

    movies, ratings = load_data()

    movies = prepare_movies(movies)

    _, tfidf_matrix = build_content_model(
        movies
    )

    valid_users = get_valid_users(
        ratings
    )

    print(
        f"\nUsers with at least "
        f"{MIN_RATINGS} ratings: "
        f"{len(valid_users):,}"
    )

    # Select random users
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

    precision_scores = []
    recall_scores = []

    evaluated_users = 0

    for i, user_id in enumerate(
        selected_users,
        start=1
    ):

        train, test = train_test_split(
            ratings,
            user_id
        )

        # Movies rated >= 4 in test set
        relevant_movies = test[
            test["rating"] >= 4.0
        ]["movieId"].tolist()

        if len(relevant_movies) == 0:
            continue

        recommendations = get_recommendations(
            train,
            movies,
            tfidf_matrix
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

        precision_scores.append(
            precision
        )

        recall_scores.append(
            recall
        )

        evaluated_users += 1

        if i % 50 == 0:

            print(
                f"Processed {i}/"
                f"{len(selected_users)} users..."
            )

    # Final results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
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


if __name__ == "__main__":
    main()