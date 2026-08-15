import pandas as pd


# =========================
# 1. Load movies
# =========================

movies = pd.read_csv("ml/data/movies.csv")


# =========================
# 2. Basic cleaning
# =========================

movies = movies.drop_duplicates(subset="movieId")

movies["title"] = movies["title"].fillna("")

movies["genres"] = movies["genres"].fillna("")


# =========================
# 3. Convert genres
# =========================

movies["genres"] = movies["genres"].str.replace(
    "|",
    " ",
    regex=False
)


# =========================
# 4. Create movie features
# =========================

movies["features"] = (
    movies["title"] + " " + movies["genres"]
)


# =========================
# 5. Save processed dataset
# =========================

movies.to_csv(
    "ml/data/movies_processed.csv",
    index=False
)


print("Preprocessing completed!")

print("\nMovies:", len(movies))

print("\nSample:")
print(
    movies[
        ["movieId", "title", "genres", "features"]
    ].head()
)