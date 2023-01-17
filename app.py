import pandas as pd
import streamlit as st
import pickle
import requests

movies_list = pickle.load(open('movies_dict.pkl','rb'))
movies = pd.DataFrame(movies_list())

similarity = pickle.load(open('similarity.pkl','rb'))

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distance = similarity[movie_index]
    movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:7]

    recommended= []
    recommended_poster=[]
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended.append(movies.iloc[i[0]].title)

        #fetch poster from API
        recommended_poster.append(fetch_postre(movie_id))
    return recommended,recommended_poster

def fetch_postre(movie_id):
    respones = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=5cf78667dda0c64a44b7e3ff9e741235&language=en-US'.format(movie_id))
    data = respones.json()
    return 'https://image.tmdb.org/t/p/w500/'+ data['poster_path']

st.title("Movie Recommendation system")

selected_movies = st.selectbox(
    'How would you like to be contacted?',movies['title'].values)
if st.button('Recommend'):
    name,poster = recommend(selected_movies)

    col1, col2, col3, col4, col5, col6= st.columns(6)

    with col1:
        st.caption(name[0])
        st.image(poster[0])

    with col2:
        st.caption(name[1])
        st.image(poster[1])

    with col3:
        st.caption(name[2])
        st.image(poster[2])

    with col4:
        st.caption(name[3])
        st.image(poster[3])

    with col5:
        st.caption(name[4])
        st.image(poster[4])

    with col6:
        st.caption(name[5])
        st.image(poster[5])