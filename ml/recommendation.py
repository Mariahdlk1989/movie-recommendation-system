import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# =========================
# 1. Load processed movies
# =========================

movies = pd.read_csv(
    "ml/data/movies_processed.csv"
)


# =========================
# 2. Create TF-IDF matrix
# =========================

tfidf = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(
    movies["features"]
)

print("TF-IDF matrix shape:")
print(tfidf_matrix.shape)


# =========================
# 3. Create Nearest Neighbors model
# =========================

model = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

model.fit(tfidf_matrix)

print("Recommendation model is ready!")


# =========================
# 4. Movie index
# =========================

movie_indices = pd.Series(
    movies.index,
    index=movies["title"]
).drop_duplicates()


# =========================
# 5. Recommendation function
# =========================

def recommend_movies(
    movie_title,
    number_of_recommendations=10
):

    if movie_title not in movie_indices:
        return []

    movie_index = movie_indices[movie_title]

    movie_vector = tfidf_matrix[
        movie_index
    ]

    distances, indices = model.kneighbors(
        movie_vector,
        n_neighbors=number_of_recommendations + 1
    )

    recommendations = []

    for distance, index in zip(
        distances[0][1:],
        indices[0][1:]
    ):

        recommendations.append({
            "title": movies.iloc[index]["title"],
            "genres": movies.iloc[index]["genres"],
            "similarity": round(
                1 - distance,
                4
            )
        })

    return pd.DataFrame(
        recommendations
    )


# =========================
# 6. Test
# =========================

movie = "Toy Story (1995)"

recommendations = recommend_movies(
    movie,
    10
)

print("\n===== RECOMMENDATIONS =====")

print(
    recommendations.to_string(
        index=False
    )
)