from Training_Phase_1 import Matrix_Factorization, Factorization_Machine
from Factorization_Machine_Library import FM_Regression
import numpy as np
import pandas as pd
import joblib

Ratings = pd.read_csv("/Users/chikhoado/Desktop/PROJECTS/Movie Recommendation/MovieLens_100k_Dataset/ratings.csv")

user_movie_matrix_Tags = joblib.load ("user_movie_matrix_Tags.pkl")
extract_genre_Movies = joblib.load ("extract_genre_Movies.pkl")
convert_unique_user_ID_to_unique_index_Tags = joblib.load ("convert_unique_user_ID_to_unique_index_Tags.pkl")
convert_unique_movie_ID_to_unique_index_Tags = joblib.load ("convert_unique_movie_ID_to_unique_index_Tags.pkl")

train_user_ID_Ratings = np.array (Ratings["userId"].unique ())
train_movie_ID_Ratings = np.array (Ratings["movieId"].unique ())
train_n_user_ID_Ratings = len (train_user_ID_Ratings)
train_m_movie_ID_Ratings = len (train_movie_ID_Ratings)

train_convert_unique_user_ID_to_unique_index_Ratings = {}
train_convert_unique_movie_ID_to_unique_index_Ratings = {}
for i in range (train_n_user_ID_Ratings):
    index = train_user_ID_Ratings[i]
    train_convert_unique_user_ID_to_unique_index_Ratings[index] = i
for i in range (train_m_movie_ID_Ratings):
    index = train_movie_ID_Ratings[i]
    train_convert_unique_movie_ID_to_unique_index_Ratings[index] = i

MF = Matrix_Factorization ()
MF.user_movie_matrix_creation (False, "rating", train_user_ID_Ratings, train_movie_ID_Ratings, False, train_n_user_ID_Ratings, train_m_movie_ID_Ratings, Ratings, train_convert_unique_user_ID_to_unique_index_Ratings, train_convert_unique_movie_ID_to_unique_index_Ratings)
train_user_movie_Ratings = MF.get_user_movie_matrix ()

model = FM_Regression ()
FM = Factorization_Machine ()
FM.compressed_column_wise (True, train_user_ID_Ratings, train_movie_ID_Ratings, train_user_movie_Ratings, user_movie_matrix_Tags, extract_genre_Movies, convert_unique_user_ID_to_unique_index_Tags, convert_unique_movie_ID_to_unique_index_Tags)
FM.train_model (model)

joblib.dump (model, "Factorization_Model_Trained.pkl")
joblib.dump (FM, "FM.pkl")








# cd "/Users/chikhoado/Desktop/PROJECTS/Movie Recommendation"
# /opt/homebrew/bin/python3.12 -m venv .venv
# source .venv/bin/activate
# pip install pandas scikit-learn numpy seaborn joblib scikit-surprise "numpy<2.0.0" 
# python "/Users/chikhoado/Desktop/PROJECTS/Movie Recommendation/Training_Phase_2.py"