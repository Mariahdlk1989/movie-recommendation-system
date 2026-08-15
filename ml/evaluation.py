import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ndcg_score


MOVIES_PATH = "ml/data/movies.csv"
RATINGS_PATH = "ml/data/ratings.csv"

TOP_K = 10
MIN_RATINGS = 20
N_USERS = 500

RANDOM_STATE = 42


def load_data():
    """Load movies and ratings datasets."""

    print("Loading data...")

    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)

    print(f"Movies: {len(movies):,}")
    print(f"Ratings: {len(ratings):,}")

    return movies, ratings


def prepare_movies(movies):
    """Prepare movie features for TF-IDF."""

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
    """Build the TF-IDF content-based recommendation model."""

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
    """Return users with at least MIN_RATINGS ratings."""

    user_counts = ratings.groupby("userId").size()

    valid_users = user_counts[
        user_counts >= MIN_RATINGS
    ].index

    return valid_users


def train_test_split(ratings, user_id):
    """
    Split a user's ratings chronologically.

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

    train = user_ratings.iloc[:split_index]

    test = user_ratings.iloc[split_index:]

    return train, test


def get_recommendations(
    train,
    movies,
    tfidf_matrix
):
    """
    Generate content-based recommendations
    based on movies the user liked in the training set.
    """

    # Movies already rated by the user
    rated_movies = set(
        train["movieId"]
    )

    # Consider ratings >= 4 as liked movies
    liked_movies = train[
        train["rating"] >= 4.0
    ]

    if liked_movies.empty:
        return []

    # Map movieId -> row index
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

    # Calculate similarity between liked movies
    # and all movies
    similarity_matrix = cosine_similarity(
        tfidf_matrix[liked_indices],
        tfidf_matrix
    )

    # Average similarity across liked movies
    scores = similarity_matrix.mean(
        axis=0
    )

    # Don't recommend movies already rated
    for movie_id in rated_movies:

        if movie_id in movie_id_to_index:

            index = movie_id_to_index[
                movie_id
            ]

            scores[index] = -1

    # Get Top-K movies
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
    """
    Calculate Precision@K.

    Precision@K =
    relevant recommended movies / K
    """

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
    """
    Calculate Recall@K.

    Recall@K =
    relevant recommended movies /
    total relevant movies
    """

    relevant = set(relevant)

    if len(relevant) == 0:
        return 0.0

    recommended = recommended[:k]

    hits = len(
        set(recommended) & relevant
    )

    return hits / len(relevant)


def ndcg_at_k(
    recommended,
    relevant,
    k
):
    """
    Calculate NDCG@K.

    NDCG considers both:
    - whether recommendations are relevant
    - their ranking position
    """

    recommended = recommended[:k]

    if len(recommended) == 0:
        return 0.0

    relevant = set(relevant)

    # Binary relevance:
    # 1 = relevant
    # 0 = not relevant
    relevance_scores = [
        1 if movie_id in relevant else 0
        for movie_id in recommended
    ]

    # Ideal ranking:
    # Relevant movies should appear first
    ideal_scores = sorted(
        relevance_scores,
        reverse=True
    )

    return ndcg_score(
        [ideal_scores],
        [relevance_scores],
        k=k
    )


def main():

    print("=" * 60)
    print("MOVIE RECOMMENDATION - MODEL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------

    movies, ratings = load_data()

    # --------------------------------------------------
    # 2. Prepare movie features
    # --------------------------------------------------

    movies = prepare_movies(
        movies
    )

    # --------------------------------------------------
    # 3. Build TF-IDF model
    # --------------------------------------------------

    _, tfidf_matrix = build_content_model(
        movies
    )

    # --------------------------------------------------
    # 4. Find valid users
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
    # 5. Select users for evaluation
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
    # 6. Store evaluation metrics
    # --------------------------------------------------

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    evaluated_users = 0

    # --------------------------------------------------
    # 7. Evaluate each user
    # --------------------------------------------------

    for i, user_id in enumerate(
        selected_users,
        start=1
    ):

        # Train/Test split
        train, test = train_test_split(
            ratings,
            user_id
        )

        # Movies the user liked in test set
        relevant_movies = test[
            test["rating"] >= 4.0
        ]["movieId"].tolist()

        # Skip users without relevant test movies
        if len(relevant_movies) == 0:
            continue

        # Generate recommendations
        recommendations = get_recommendations(
            train,
            movies,
            tfidf_matrix
        )

        if not recommendations:
            continue

        # Calculate metrics
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

        # Store results
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

        # Progress
        if i % 50 == 0:

            print(
                f"Processed {i}/"
                f"{len(selected_users)} users..."
            )

    # --------------------------------------------------
    # 8. Final evaluation results
    # --------------------------------------------------

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

    mean_ndcg = np.mean(
        ndcg_scores
    )

    # --------------------------------------------------
    # 9. Print metrics
    # --------------------------------------------------

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