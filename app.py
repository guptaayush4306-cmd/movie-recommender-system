import streamlit as st
import pandas as pd
import ast
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# PAGE SETTINGS
# -------------------------------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():

    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    movies = movies.merge(credits, on="title")

    movies = movies[
        ["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]
    ]

    movies.dropna(inplace=True)

    return movies


movies = load_data()

# -------------------------------------------------
# PREPROCESSING FUNCTIONS
# -------------------------------------------------
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i["name"])
    return L


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i["job"] == "Director":
            L.append(i["name"])
    return L


def collapse(L):
    return [i.replace(" ", "") for i in L]


# -------------------------------------------------
# DATA PREPROCESSING
# -------------------------------------------------
movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)
movies["cast"] = movies["cast"].apply(convert)
movies["crew"] = movies["crew"].apply(fetch_director)
movies["overview"] = movies["overview"].apply(lambda x: x.split())

for col in ["genres", "keywords", "cast", "crew"]:
    movies[col] = movies[col].apply(collapse)

movies["tags"] = (
    movies["overview"]
    + movies["genres"]
    + movies["keywords"]
    + movies["cast"]
    + movies["crew"]
)

new_df = movies[["movie_id", "title", "tags"]].copy()

new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x))
new_df["tags"] = new_df["tags"].apply(lambda x: x.lower())

# -------------------------------------------------
# CREATE SIMILARITY
# -------------------------------------------------
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(new_df["tags"]).toarray()

similarity = cosine_similarity(vectors)

# -------------------------------------------------
# POSTER API
# -------------------------------------------------
API_KEY = "786a7ba4aec540b31e9e996558061c84"


def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"

        response = requests.get(url, timeout=10)
        data = response.json()

        poster_path = data.get("poster_path")

        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path
        else:
            return "https://via.placeholder.com/500x750?text=No+Poster"

    except:
        return "https://via.placeholder.com/500x750?text=Error"

# -------------------------------------------------
# RECOMMEND FUNCTION
# -------------------------------------------------
def recommend(movie):
    index = new_df[new_df["title"] == movie].index[0]
    distances = similarity[index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1],
    )[1:6]

    names = []
    posters = []

    for i in movies_list:
        movie_id = new_df.iloc[i[0]].movie_id
        names.append(new_df.iloc[i[0]].title)
        posters.append(fetch_poster(movie_id))

    return names, posters


# -------------------------------------------------
# UI
# -------------------------------------------------
selected_movie = st.selectbox(
    "Select a movie:", new_df["title"].values
)

if st.button("Recommend 🎥"):

    names, posters = recommend(selected_movie)

    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            if posters[i]:
                st.image(posters[i])
            st.caption(names[i])
