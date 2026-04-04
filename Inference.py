import numpy as np
import joblib
import re

model = joblib.load ("Factorization_Model_Trained.pkl")
FM = joblib.load ("FM.pkl")
unique_movie_id = joblib.load ("unique_movie_id.pkl")
genre_dict = joblib.load ("unique_movie_genre.pkl")
extract_title_Movies = joblib.load ("extract_title_Movies.pkl")
extract_imdb_Links = joblib.load ("extract_imdb_Links.pkl")
extract_tmdb_link = joblib.load ("extract_tmdb_links.pkl")

def print_recommended_movie (Top_K_Movie):

    print ("\n--------------------- Follow Instructions ---------------------")

    user_id = np.int64 (input ("\n+ Type your User ID (if you dont have, better enter -1): ").strip ())

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
    
    movie_genre_idx = input ("Select your favourite genres (use comma to separate choices): ").strip ()
    movie_genre_idx = re.split (r',', movie_genre_idx)
    movie_genre_idx = [int (temp.strip ()) for temp in movie_genre_idx]

    print ("\n--------------------- System is processing, please wait... --------------------- ")
    
    out_of_range = [idx for idx in movie_genre_idx if not (1 <= idx <= 20)]
    if out_of_range:
        temp = ', '.join (map (str, out_of_range))
        print (f"\n+ Your list contains out of selection range (including: {temp}) so we'll erase them...")
        movie_genre_idx = [idx for idx in movie_genre_idx if 1 <= idx <= 20]
    
    movie_genre = [unique_movie_genre[idx - 1] for idx in movie_genre_idx]
    X = []

    for movie_id in unique_movie_id:

        datapoint = {
                "USER_ID": user_id, 
                "MOVIE_ID": movie_id,
        }
        for genre in movie_genre:
            datapoint[genre] = genre

        X.append (datapoint)

    X = FM.CSC_Inference (X)
    rating_score = model.predict (X)
    combined = zip (rating_score, unique_movie_id)
    sort_rating_score = sorted (combined)

    print ("\n+", end = '')
    for i in range (Top_K_Movie):
        pair = sort_rating_score[i]
        movie_id = pair[1]
        title = extract_title_Movies[movie_id]  

        if i == 0:
            pad = ' '
        else:
            pad = '  '
        print (f'{pad}[Top {i + 1}] --- Title    : {title}')

        if movie_id in extract_imdb_Links:
            imdb = extract_imdb_Links[movie_id]
            if len (str (imdb)) == 6:
                print (f'              IMDB Link: https://www.imdb.com/title/tt0{imdb}/')
            else:
                print (f'              IMDB Link: https://www.imdb.com/title/tt{imdb}/')
        else:
            print (f'               IMDB Link: Sorry! We dont have IMDB Link for this movie :( ')

        if movie_id in extract_tmdb_link:
            tmdb = extract_tmdb_link[movie_id]
            print (f'              TMDB Link: https://www.themoviedb.org/movie/{tmdb}/')
        else:
            print (f'              TMDB Link: Sorry! We dont have TMDB Link for this movie :( ')

        print ('')
        

try:
    top_k_movie = 5
    print_recommended_movie (top_k_movie)

except ValueError as e:
    print ("\n----------- ERROR: please type only integer AND not leave it blank -----------")




# cd "/Users/chikhoado/Desktop/PROJECTS/Movie Recommendation"
# /opt/homebrew/bin/python3.12 -m venv .venv
# source .venv/bin/activate
# pip install pandas scikit-learn numpy seaborn joblib scikit-surprise "numpy<2.0.0"
# python "/Users/chikhoado/Desktop/PROJECTS/Movie Recommendation/Inference.py"