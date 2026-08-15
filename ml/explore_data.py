import pandas as pd

# =========================
# 1. Load datasets
# =========================

movies = pd.read_csv("ml/data/movies.csv")
ratings = pd.read_csv("ml/data/ratings.csv")


# =========================
# 2. Basic information
# =========================

print("=" * 50)
print("MOVIE RECOMMENDATION SYSTEM")
print("=" * 50)

print("\n===== DATASET SHAPES =====")

print("Number of movies:", len(movies))
print("Number of ratings:", len(ratings))
print("Number of users:", ratings["userId"].nunique())


# =========================
# 3. Movies dataset
# =========================

print("\n===== MOVIES =====")

print(movies.head())

print("\nMovie columns:")
print(movies.columns.tolist())


# =========================
# 4. Ratings dataset
# =========================

print("\n===== RATINGS =====")

print(ratings.head())

print("\nRating columns:")
print(ratings.columns.tolist())


# =========================
# 5. Missing values
# =========================

print("\n===== MISSING VALUES =====")

print("\nMovies:")
print(movies.isnull().sum())

print("\nRatings:")
print(ratings.isnull().sum())


# =========================
# 6. Rating statistics
# =========================

print("\n===== RATING STATISTICS =====")

print(ratings["rating"].describe())


# =========================
# 7. Rating distribution
# =========================

print("\n===== RATING DISTRIBUTION =====")

print(
    ratings["rating"]
    .value_counts()
    .sort_index()
)


# =========================
# 8. Most rated movies
# =========================

print("\n===== MOST RATED MOVIES =====")

rating_counts = ratings["movieId"].value_counts()

top_movies = (
    rating_counts
    .head(10)
    .rename_axis("movieId")
    .reset_index(name="rating_count")
)

top_movies = top_movies.merge(
    movies,
    on="movieId"
)

print(
    top_movies[
        ["movieId", "title", "rating_count"]
    ].to_string(index=False)
)
# =========================
# 9. User activity
# =========================

user_rating_counts = ratings["userId"].value_counts()

print("\n===== USER ACTIVITY =====")

print("Average ratings per user:",
      user_rating_counts.mean())

print("Median ratings per user:",
      user_rating_counts.median())

print("Maximum ratings by one user:",
      user_rating_counts.max())


# =========================
# 10. Movie popularity
# =========================

movie_rating_counts = ratings["movieId"].value_counts()

print("\n===== MOVIE POPULARITY =====")

print("Average ratings per movie:",
      movie_rating_counts.mean())

print("Median ratings per movie:",
      movie_rating_counts.median())

print("Maximum ratings for one movie:",
      movie_rating_counts.max())


# =========================
# 11. Rating coverage
# =========================

print("\n===== DATASET SUMMARY =====")

print("Users:", ratings["userId"].nunique())
print("Movies rated:", ratings["movieId"].nunique())
print("Total movies:", movies["movieId"].nunique())
print("Total ratings:", len(ratings))