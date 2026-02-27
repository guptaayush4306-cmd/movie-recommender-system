import streamlit as st
import pandas as pd
import numpy as np
import ast
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")

# -------------------------------
# LOAD DATASETS
# -------------------------------
@st.cache_data
def load_data():
    movies_url = "https://raw.githubusercontent.com/omkarpathak/py-movie-recommender-system/master/tmdb_5000_movies.csv"
    credits_url = "https://raw.githubusercontent.com/omkarpathak/py-movie-recommender-system/master/tmdb_5000_credits.csv"

    movies = pd.read_csv(movies_url)
    credits = pd.read_csv(credits_url)

    return movies, credits

movies, credits = load_data()

movies = movies.merge(credits, on="title")

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.dropna(inplace=True)

# -------------------------------
# DATA PREPROCESSING
# -------------------------------

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert)

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].apply(lambda x: x.split())

def collapse(L):
    return [i.replace(" ", "") for i in L]

for col in ['genres','keywords','cast','crew']:
    movies[col] = movies[col].apply(collapse)

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id','title','tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# -------------------------------
# CREATE SIMILARITY MATRIX
# -------------------------------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

similarity = cosine_similarity(vectors)

# -------------------------------
# TMDB POSTER FUNCTION
# -------------------------------
API_KEY = "YOUR_API_KEY"

def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
        headers = {"User-Agent": "Mozilla/5.0"}
        data = requests.get(url, headers=headers, timeout=10).json()

        poster_path = data.get("poster_path")

        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            return "https://via.placeholder.com/500x750?text=No+Poster"
    except:
        return "https://via.placeholder.com/500x750?text=Error"

# -------------------------------
# RECOMMENDATION FUNCTION
# -------------------------------
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movies_list:
        movie_id = new_df.iloc[i[0]].movie_id
        recommended_movies.append(new_df.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters

# -------------------------------
# UI
# -------------------------------
selected_movie = st.selectbox(
    "Select a movie:",
    new_df['title'].values
)

if st.button("Recommend 🎥"):

    with st.spinner("Finding similar movies... 🍿"):
        names, posters = recommend(selected_movie)

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(posters[0])
            st.caption(names[0])

        with col2:
            st.image(posters[1])
            st.caption(names[1])

        with col3:
            st.image(posters[2])
            st.caption(names[2])

        with col4:
            st.image(posters[3])
            st.caption(names[3])

        with col5:
            st.image(posters[4])
            st.caption(names[4])