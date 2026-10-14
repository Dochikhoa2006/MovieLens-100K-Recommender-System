import numpy as np
import joblib
from rated_movies import load_rated_movies
from recommendation_candidates import build_candidates

model = joblib.load ("Factorization_Model_Trained.pkl")
FM = joblib.load ("FM.pkl")
unique_movie_id = joblib.load ("unique_movie_id.pkl")
genre_dict = joblib.load ("unique_movie_genre.pkl")
genres_by_movie = joblib.load ("extract_genre_Movies.pkl")
extract_title_Movies = joblib.load ("extract_title_Movies.pkl")
extract_imdb_Links = joblib.load ("extract_imdb_Links.pkl")
extract_tmdb_link = joblib.load ("extract_tmdb_links.pkl")
rated_movies = load_rated_movies()

def print_recommended_movie (Top_K_Movie):

    print ("\n--------------------- Follow Instructions ---------------------")

    user_id = np.int64 (input ("\n+ Type your User ID (if you dont have, better enter -1): ").strip ())

    if user_id != -1 and user_id not in rated_movies:
        print ("\n+ Unknown User ID. Enter a known ID or -1 if you are a new user.")
        return

    print ("\n+ List of Movie Genres: ", end = '')

    unique_movie_genre = [key for key, value in genre_dict.items ()]
    for index in range (len (unique_movie_genre)):
        uni = unique_movie_genre[index]

        if index == 0:
            print (f'[{index + 1}]: ', uni, sep = '')
        else:
            if uni == "(no genres listed)":
                uni = "I dont interested at any above Movie Genres"
            print (f'                        [{index + 1}]: ', uni, sep = '')
    
    genre_choices = input ("Select your favourite genres (use comma to separate choices): ")
    movie_genre_idx = []
    invalid_choices = []
    for choice in genre_choices.split (","):
        choice = choice.strip ()
        try:
            movie_genre_idx.append (int (choice))
        except ValueError:
            invalid_choices.append (choice or "(blank)")

    if invalid_choices:
        print (f"\n+ Ignoring non-numeric choices: {', '.join (invalid_choices)}")

    print ("\n--------------------- System is processing, please wait... --------------------- ")
    
    out_of_range = [idx for idx in movie_genre_idx if not (1 <= idx <= len (unique_movie_genre))]
    if out_of_range:
        temp = ', '.join (map (str, out_of_range))
        print (f"\n+ Your list contains out of selection range (including: {temp}) so we'll erase them...")
        movie_genre_idx = [idx for idx in movie_genre_idx if 1 <= idx <= len (unique_movie_genre)]

    if not movie_genre_idx:
        print ("\n+ Please select at least one valid genre.")
        return
    
    movie_genre = [unique_movie_genre[idx - 1] for idx in movie_genre_idx]
    already_rated = rated_movies.get(user_id, set())
    X, candidate_movie_ids = build_candidates(
        user_id, movie_genre, unique_movie_id, genres_by_movie, already_rated
    )

    if not X:
        print ("\n+ No unrated movies match the selected genres.")
        return

    X = FM.CSC_Inference (X)
    rating_score = model.predict (X)
    combined = zip (rating_score, candidate_movie_ids)
    sort_rating_score = sorted (combined, reverse=True)

    print ("\n+", end = '')
    for i, pair in enumerate (sort_rating_score[:Top_K_Movie]):
        score = pair[0]
        movie_id = pair[1]
        title = extract_title_Movies[movie_id]  

        if i == 0:
            pad = ' '
        else:
            pad = '  '
        print (f'{pad}[Top {i + 1}] --- Title    : {title}')
        print (f'              Predicted rating: {score:.2f}/5')

        if movie_id in extract_imdb_Links:
            imdb = str (extract_imdb_Links[movie_id]).zfill (7)
            print (f'              IMDB Link: https://www.imdb.com/title/tt{imdb}/')
        else:
            print (f'               IMDB Link: Sorry! We dont have IMDB Link for this movie :( ')

        if movie_id in extract_tmdb_link:
            tmdb = extract_tmdb_link[movie_id]
            print (f'              TMDB Link: https://www.themoviedb.org/movie/{tmdb}/')
        else:
            print (f'              TMDB Link: Sorry! We dont have TMDB Link for this movie :( ')

        print ('')
        

if __name__ == "__main__":
    try:
        top_k_movie = 5
        print_recommended_movie (top_k_movie)
    except ValueError:
        print ("\n----------- ERROR: please type only integer AND not leave it blank -----------")
