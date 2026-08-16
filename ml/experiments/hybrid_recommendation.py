import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# =========================
# 1. Load processed movies
# =========================

movies = pd.read_csv(
    "ml/data/movies_with_stats.csv"
)


# =========================
# 2. Handle missing values
# =========================

movies["features"] = (
    movies["features"]
    .fillna("")
)

movies["average_rating"] = (
    movies["average_rating"]
    .fillna(0)
)

movies["rating_count"] = (
    movies["rating_count"]
    .fillna(0)
)


# =========================
# 3. Create TF-IDF matrix
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
# 4. Create Nearest Neighbors model
# =========================

model = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

model.fit(tfidf_matrix)

print("Recommendation model is ready!")


# =========================
# 5. Create movie index
# =========================

movie_indices = pd.Series(
    movies.index,
    index=movies["title"]
).drop_duplicates()


# =========================
# 6. Bayesian Weighted Rating
# =========================

# Global average rating
global_mean = movies["average_rating"].mean()

# Minimum number of ratings
# required to trust a movie's average rating
m = 100

movies["weighted_rating"] = (
    (
        movies["rating_count"]
        /
        (movies["rating_count"] + m)
    )
    *
    movies["average_rating"]
    +
    (
        m
        /
        (movies["rating_count"] + m)
    )
    *
    global_mean
)


# =========================
# 7. Normalize weighted rating
# =========================

min_rating = movies["weighted_rating"].min()
max_rating = movies["weighted_rating"].max()

if max_rating > min_rating:

    movies["rating_score"] = (
        (
            movies["weighted_rating"]
            - min_rating
        )
        /
        (
            max_rating
            - min_rating
        )
    )

else:

    movies["rating_score"] = 0


# =========================
# 8. Recommendation function
# =========================

def recommend_movies(
    movie_title,
    number_of_recommendations=10
):

    # -------------------------
    # Check movie
    # -------------------------

    if movie_title not in movie_indices:

        print(
            f"Movie not found: {movie_title}"
        )

        return pd.DataFrame()


    # -------------------------
    # Get movie index
    # -------------------------

    movie_index = movie_indices[
        movie_title
    ]


    # -------------------------
    # Get movie vector
    # -------------------------

    movie_vector = tfidf_matrix[
        movie_index
    ]


    # -------------------------
    # Find similar movies
    # -------------------------

    distances, indices = model.kneighbors(
        movie_vector,
        n_neighbors=number_of_recommendations + 20
    )


    recommendations = []


    # -------------------------
    # Calculate hybrid score
    # -------------------------

    for distance, index in zip(
        distances[0][1:],
        indices[0][1:]
    ):

        # Convert cosine distance
        # to cosine similarity

        similarity = 1 - distance


        # Get Bayesian rating score

        rating_score = movies.iloc[
            index
        ]["rating_score"]


        # -------------------------
        # Hybrid score
        # -------------------------

        hybrid_score = (
            0.75 * similarity
            +
            0.25 * rating_score
        )


        # -------------------------
        # Store recommendation
        # -------------------------

        recommendations.append({

            "title": movies.iloc[
                index
            ]["title"],

            "genres": movies.iloc[
                index
            ]["genres"],

            "similarity": round(
                similarity,
                4
            ),

            "average_rating": round(
                movies.iloc[index][
                    "average_rating"
                ],
                2
            ),

            "weighted_rating": round(
                movies.iloc[index][
                    "weighted_rating"
                ],
                2
            ),

            "rating_count": int(
                movies.iloc[index][
                    "rating_count"
                ]
            ),

            "hybrid_score": round(
                hybrid_score,
                4
            )
        })


    # -------------------------
    # Sort recommendations
    # -------------------------

    recommendations = sorted(
        recommendations,
        key=lambda x: x["hybrid_score"],
        reverse=True
    )


    # -------------------------
    # Return top results
    # -------------------------

    return pd.DataFrame(
        recommendations[
            :number_of_recommendations
        ]
    )


# =========================
# 9. Test recommendation
# =========================

movie = "Toy Story (1995)"


recommendations = recommend_movies(
    movie,
    10
)


# =========================
# 10. Display results
# =========================

print(
    "\n===== HYBRID RECOMMENDATIONS ====="
)


if not recommendations.empty:

    print(
        recommendations.to_string(
            index=False
        )
    )