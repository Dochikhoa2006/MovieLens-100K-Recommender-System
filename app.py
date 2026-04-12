import streamlit as st
import joblib

st.set_page_config(page_title="Movie Recommender", layout="centered")

@st.cache_resource
def load_assets():
    model = joblib.load("Factorization_Model_Trained.pkl")
    FM = joblib.load("FM.pkl")
    unique_movie_id = joblib.load("unique_movie_id.pkl")
    genre_dict = joblib.load("unique_movie_genre.pkl")
    extract_title_Movies = joblib.load("extract_title_Movies.pkl")
    extract_imdb_Links = joblib.load("extract_imdb_Links.pkl")
    extract_tmdb_link = joblib.load("extract_tmdb_links.pkl")
    return model, FM, unique_movie_id, genre_dict, extract_title_Movies, extract_imdb_Links, extract_tmdb_link

model, FM, unique_movie_id, genre_dict, extract_title_Movies, extract_imdb_Links, extract_tmdb_link = load_assets()

st.title("🎬 Movie Recommendation System")
st.markdown("Enter your ID and favorite genres to get personalized top picks.")

with st.container():
    user_id = st.number_input("Type your User ID (Enter -1 if new user):", value=-1, step=1)
    
    unique_movie_genre = [key for key in genre_dict.keys()]
    display_genres = [g if g != "(no genres listed)" else "None of the above" for g in unique_movie_genre]
    
    selected_genres_display = st.multiselect(
        "Select your favourite genres:",
        options=display_genres
    )

    selected_genres = [unique_movie_genre[display_genres.index(g)] for g in selected_genres_display]
    
    top_k = st.slider("Number of movies to recommend:", 1, 20, 5)

if st.button("Get Recommendations"):
    if not selected_genres:
        st.warning("Please select at least one genre.")
    else:
        with st.spinner("Processing..."):
            X = []
            for movie_id in unique_movie_id:
                datapoint = {"USER_ID": user_id, "MOVIE_ID": movie_id}
                for genre in selected_genres:
                    datapoint[genre] = genre
                X.append(datapoint)

            X_transformed = FM.CSC_Inference(X)
            rating_score = model.predict(X_transformed)
            
            combined = sorted(zip(rating_score, unique_movie_id), reverse=True) 
            
            st.success(f"Top {top_k} Recommendations for you:")
            
            for i in range(top_k):
                score, movie_id = combined[i]
                title = extract_title_Movies.get(movie_id, "Unknown Title")
                
                with st.expander(f"Top {i+1}: {title}"):
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



# cd "/Users/chikhoado/Desktop/PROJECTS/Movie Recommendation"
# /opt/homebrew/bin/python3.12 -m venv .venv
# source .venv/bin/activate
# pip install pandas scikit-learn numpy seaborn joblib regex scikit-surprise streamlit
# streamlit run "/Users/chikhoado/Desktop/PROJECTS/Movie Recommendation/Inference_with_Streamlit.py"