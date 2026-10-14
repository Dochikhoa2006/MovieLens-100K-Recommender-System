import streamlit as st
import joblib
from pathlib import Path
from rated_movies import load_rated_movies
from recommendation_candidates import build_candidates

st.set_page_config (page_title = "Movie Recommender", layout = "centered")
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_assets ():
    model = joblib.load (BASE_DIR / "Factorization_Model_Trained.pkl")
    FM = joblib.load (BASE_DIR / "FM.pkl")
    unique_movie_id = joblib.load (BASE_DIR / "unique_movie_id.pkl")
    genre_dict = joblib.load (BASE_DIR / "unique_movie_genre.pkl")
    genres_by_movie = joblib.load (BASE_DIR / "extract_genre_Movies.pkl")
    extract_title_Movies = joblib.load (BASE_DIR / "extract_title_Movies.pkl")
    extract_imdb_Links = joblib.load (BASE_DIR / "extract_imdb_Links.pkl")
    extract_tmdb_link = joblib.load (BASE_DIR / "extract_tmdb_links.pkl")
    return model, FM, unique_movie_id, genre_dict, genres_by_movie, extract_title_Movies, extract_imdb_Links, extract_tmdb_link

model, FM, unique_movie_id, genre_dict, genres_by_movie, extract_title_Movies, extract_imdb_Links, extract_tmdb_link = load_assets ()

@st.cache_data
def get_rated_movies():
    return load_rated_movies()

st.title ("🎬 Movie Recommendation System")
st.markdown ("Enter your ID and favorite genres to get personalized top picks.")

with st.container ():
    user_id = st.number_input ("Type your User ID (Enter -1 if new user):", value = -1, step = 1)
    
    unique_movie_genre = [key for key in genre_dict.keys ()]
    display_genres = [g if g != "(no genres listed)" else "None of the above" for g in unique_movie_genre]
    
    selected_genres_display = st.multiselect(
        "Select your favourite genres:",
        options = display_genres
    )

    selected_genres = [unique_movie_genre[display_genres.index (g)] for g in selected_genres_display]
    
    top_k = st.slider ("Number of movies to recommend:", 1, 20, 5)

if st.button ("Get Recommendations"):
    rated_movies = get_rated_movies()
    if user_id != -1 and user_id not in rated_movies:
        st.warning ("Unknown User ID. Enter a known ID or -1 if you are a new user.")
    elif not selected_genres:
        st.warning ("Please select at least one genre.")
    else:
        with st.spinner ("Processing..."):
            already_rated = rated_movies.get(user_id, set())
            X, candidate_movie_ids = build_candidates(
                user_id, selected_genres, unique_movie_id, genres_by_movie, already_rated
            )

            if not X:
                st.info("No unrated movies match the selected genres.")
                st.stop()

            X_transformed = FM.CSC_Inference(X)
            rating_score = model.predict(X_transformed)
            
            combined = sorted(zip(rating_score, candidate_movie_ids), reverse=True)
            recommendations = combined[:top_k]
            
            st.success (f"Top {len(recommendations)} Recommendations for you:")
            
            for i, (score, movie_id) in enumerate(recommendations):
                title = extract_title_Movies.get(movie_id, "Unknown Title")
                
                with st.expander(f"Top {i+1}: {title} — predicted rating {score:.2f}/5"):
                    if movie_id in extract_imdb_Links:
                        imdb = str(extract_imdb_Links[movie_id]).zfill(7)
                        st.markdown(f"[🔗 View on IMDB](https://www.imdb.com/title/tt{imdb}/)")
                    else:
                        st.write("IMDB Link: Not available")
                    
                    if movie_id in extract_tmdb_link:
                        tmdb = extract_tmdb_link[movie_id]
                        st.markdown(f"[🔗 View on TMDB](https://www.themoviedb.org/movie/{tmdb}/)")
                    else:
                        st.write("TMDB Link: Not available")
