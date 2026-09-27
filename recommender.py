import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")

movies["features"] = (
    movies["genre"].fillna("") + " " +
    movies["overview"].fillna("")
)

vectorizer = TfidfVectorizer(stop_words="english")
feature_matrix = vectorizer.fit_transform(movies["features"])

similarity = cosine_similarity(feature_matrix)


def recommend_movies(movie_title, number_of_movies=5):

    movie_index = movies[
        movies["title"].str.lower() == movie_title.lower()
    ].index

    if len(movie_index) == 0:
        return []

    movie_index = movie_index[0]

    scores = list(enumerate(similarity[movie_index]))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for index, score in scores[1:number_of_movies + 1]:

        recommendations.append({
            "title": movies.iloc[index]["title"],
            "poster": movies.iloc[index]["poster"]
        })

    return recommendations