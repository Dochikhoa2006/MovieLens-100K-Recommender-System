import argparse
from pathlib import Path
import numpy as np
import joblib
from rated_movies import load_rated_movies
from recommendation_candidates import build_candidates

BASE_DIR = Path(__file__).resolve().parent


def load_assets():
    model = joblib.load(BASE_DIR / "Factorization_Model_Trained.pkl")
    FM = joblib.load(BASE_DIR / "FM.pkl")
    unique_movie_id = joblib.load(BASE_DIR / "unique_movie_id.pkl")
    genre_dict = joblib.load(BASE_DIR / "unique_movie_genre.pkl")
    genres_by_movie = joblib.load(BASE_DIR / "extract_genre_Movies.pkl")
    extract_title_Movies = joblib.load(BASE_DIR / "extract_title_Movies.pkl")
    extract_imdb_Links = joblib.load(BASE_DIR / "extract_imdb_Links.pkl")
    extract_tmdb_link = joblib.load(BASE_DIR / "extract_tmdb_links.pkl")
    rated_movies = load_rated_movies(BASE_DIR / "MovieLens_100k_Dataset/ratings.csv")
    return (model, FM, unique_movie_id, genre_dict, genres_by_movie,
            extract_title_Movies, extract_imdb_Links, extract_tmdb_link, rated_movies)


def print_genres(genres):
    for index, genre in enumerate(genres, start=1):
        if genre == "(no genres listed)":
            genre = "None of the above"
        print(f"{index}: {genre}")


def print_recommended_movie (Top_K_Movie, user_id=None, genre_choices=None):
    (model, FM, unique_movie_id, genre_dict, genres_by_movie,
     extract_title_Movies, extract_imdb_Links, extract_tmdb_link, rated_movies) = load_assets()

    if user_id is None or genre_choices is None:
        print ("\n--------------------- Follow Instructions ---------------------")

    if user_id is None:
        user_id = np.int64 (input ("\n+ Type your User ID (if you dont have, better enter -1): ").strip ())

    if user_id != -1 and user_id not in rated_movies:
        print ("\n+ Unknown User ID. Enter a known ID or -1 if you are a new user.")
        return

    unique_movie_genre = [key for key, value in genre_dict.items ()]
    if genre_choices is None:
        print ("\n+ List of Movie Genres:")
        print_genres(unique_movie_genre)
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
    parser = argparse.ArgumentParser(description="Recommend movies by genre.")
    parser.add_argument("--user-id", type=int, help="Known user ID, or -1 for a new user")
    parser.add_argument("--genres", help="Comma-separated genre numbers shown in the interactive menu")
    parser.add_argument("--top-k", type=int, default=5, help="Number of recommendations (default: 5)")
    parser.add_argument("--list-genres", action="store_true", help="Show genre numbers and exit")
    args = parser.parse_args()
    if args.list_genres:
        print_genres(joblib.load(BASE_DIR / "unique_movie_genre.pkl").keys())
        raise SystemExit(0)
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    try:
        print_recommended_movie (args.top_k, args.user_id, args.genres)
    except ValueError:
        print ("\n----------- ERROR: please type only integer AND not leave it blank -----------")
