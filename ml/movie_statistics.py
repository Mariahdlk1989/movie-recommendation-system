import pandas as pd


# =========================
# 1. Load data
# =========================

movies = pd.read_csv(
    "ml/data/movies_processed.csv"
)

ratings = pd.read_csv(
    "ml/data/ratings.csv"
)


# =========================
# 2. Calculate statistics
# =========================

movie_stats = (
    ratings
    .groupby("movieId")
    .agg(
        rating_count=("rating", "count"),
        average_rating=("rating", "mean")
    )
    .reset_index()
)


# =========================
# 3. Merge with movies
# =========================

movies = movies.merge(
    movie_stats,
    on="movieId",
    how="left"
)


# =========================
# 4. Fill missing values
# =========================

movies["rating_count"] = (
    movies["rating_count"]
    .fillna(0)
)

movies["average_rating"] = (
    movies["average_rating"]
    .fillna(0)
)


# =========================
# 5. Save
# =========================

movies.to_csv(
    "ml/data/movies_with_stats.csv",
    index=False
)


# =========================
# 6. Show results
# =========================

print("Movie statistics created!")

print(
    movies[
        [
            "movieId",
            "title",
            "rating_count",
            "average_rating"
        ]
    ]
    .sort_values(
        "rating_count",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)