from Factorization_Machine_Library import FM_Regression
from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer
from scipy import sparse
from surprise import Dataset, Reader, SVDpp
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import re


Links = pd.read_csv("/Users/chikhoado/Desktop/PROJECTS/Movie Recommendation/MovieLens_100k_Dataset/links.csv")
Movies = pd.read_csv("/Users/chikhoado/Desktop/PROJECTS/Movie Recommendation/MovieLens_100k_Dataset/movies.csv")
Ratings = pd.read_csv("/Users/chikhoado/Desktop/PROJECTS/Movie Recommendation/MovieLens_100k_Dataset/ratings.csv")
Tags = pd.read_csv("/Users/chikhoado/Desktop/PROJECTS/Movie Recommendation/MovieLens_100k_Dataset/tags.csv")

cross_validation = KFold (n_splits = 10, shuffle = True, random_state = 150)
train_test_split_ratings = cross_validation.split (Ratings)
RMSE_Matrix_Factorization = []
RMSE_Factorization_Machine = []

class Matrix_Factorization:
    def __init__ (self):
        self.user_movie_matrix = None

    def user_movie_matrix_creation (self, check, string, valid_user_ID_Ratings, valid_movie_ID_Ratings, bool, n, m, train_Ratings, train_convert_unique_user_ID_to_unique_index_Ratings, train_convert_unique_movie_ID_to_unique_index_Ratings):
        
        if not check:
            self.user_movie_matrix = np.zeros ((n, m))
        else:
            self.user_movie_matrix = np.empty ((n, m), dtype = object)
            for i in range (n):
                for j in range (m):
                    self.user_movie_matrix[i, j] = []

        user_ID = train_Ratings["userId"].values
        movie_ID = train_Ratings["movieId"].values
        rating = train_Ratings[string].values
        timestamp = train_Ratings["timestamp"].values

        bool = bool
        timestamp_record = {}

        for index in train_Ratings.index:
            user_id = user_ID[index] 
            movie_id = movie_ID[index]

            if bool == True and not (user_id in valid_user_ID_Ratings and movie_id in valid_movie_ID_Ratings):
                continue

            rating_value = rating[index]
            timestamp_value = timestamp[index]
            interaction = str (user_id) + '-' + str (movie_id)

            if (not check) and interaction in timestamp_record and timestamp_value <= timestamp_record[interaction]:
                continue

            i = train_convert_unique_user_ID_to_unique_index_Ratings[user_id]
            j = train_convert_unique_movie_ID_to_unique_index_Ratings[movie_id]
            if not check:   
                timestamp_record[interaction] = timestamp_value
                self.user_movie_matrix[i, j] = rating_value
            else:
                self.user_movie_matrix[i, j].append (rating_value)

    def get_user_movie_matrix (self):
        return self.user_movie_matrix 

    def train_model (self, model_train, train_user_ID_Ratings, train_movie_ID_Ratings):
        change_format = []
        for i in range (len (self.user_movie_matrix)):
            for j in range (len (self.user_movie_matrix[i])):
                if self.user_movie_matrix[i, j] != 0:
                    temp = (train_user_ID_Ratings[i], train_movie_ID_Ratings[j], self.user_movie_matrix[i, j])
                    change_format.append (temp)
        
        reader = Reader (rating_scale = (0, 5))
        df = pd.DataFrame (change_format, columns = ("userId", "movieId", "rating"))
        dataset = Dataset.load_from_df (df, reader)
        X_train = dataset.build_full_trainset ()

        model_train.fit (X_train)

    def RMSE_cross_validation_record (self, model_test, test_user_ID_Ratings, test_movie_ID_Ratings):
        change_format = []
        for i in range (len (self.user_movie_matrix)):
            for j in range (len (self.user_movie_matrix[i])):
                if self.user_movie_matrix[i, j] != 0:
                    temp = (test_user_ID_Ratings[i], test_movie_ID_Ratings[j], self.user_movie_matrix[i, j])
                    change_format.append (temp)

        sum_square = 0
        for tup in change_format:
            prediction_unformatted = model_test.predict (tup[0], tup[1])
            prediction = prediction_unformatted.est
            observation = tup[2]
            sum_square += (prediction - observation) ** 2
        
        RMSE = np.sqrt (sum_square / len (change_format))
        RMSE_Matrix_Factorization.append (RMSE)

class Factorization_Machine:
    def __init__ (self):
        self.Y = None
        self.CSC = None
        self.dict_vectorized = DictVectorizer (sparse = True)

    def compressed_column_wise (self, boolean, train_user_ID_Ratings, train_movie_ID_Ratings, train_user_movie_Ratings, user_movie_matrix_Tags, extract_genre_Movies, convert_unique_user_ID_to_unique_index_Tags, convert_unique_movie_ID_to_unique_index_Tags):
        
        self.Y = []
        prepare_row_wise = []

        for i in range (len (train_user_movie_Ratings)):
            for j in range (len (train_user_movie_Ratings[i])):
                
                rating_score = train_user_movie_Ratings[i, j]
                if rating_score != 0:

                    user_id_rating_temp = train_user_ID_Ratings[i]
                    movie_id_rating_temp = train_movie_ID_Ratings[j]

                    datapoint = {
                            "USER_ID": user_id_rating_temp, 
                            "MOVIE_ID": movie_id_rating_temp,
                    }

                    if movie_id_rating_temp in extract_genre_Movies:
                        for characteristic in extract_genre_Movies[movie_id_rating_temp]:
                            datapoint[characteristic] = characteristic
                    
                    if (user_id_rating_temp in convert_unique_user_ID_to_unique_index_Tags) and (movie_id_rating_temp in convert_unique_movie_ID_to_unique_index_Tags):
                        user_id_tag_temp = convert_unique_user_ID_to_unique_index_Tags[user_id_rating_temp]
                        movie_id_tag_temp = convert_unique_movie_ID_to_unique_index_Tags[movie_id_rating_temp]
                        list_tag = user_movie_matrix_Tags[user_id_tag_temp, movie_id_tag_temp]

                        if list_tag:
                            for tag in list_tag:
                                datapoint[tag] = tag

                    prepare_row_wise.append (datapoint)
                    self.Y.append (rating_score)

        if boolean:
            compressed_row_wise = self.dict_vectorized.fit_transform (prepare_row_wise)
        else:
            compressed_row_wise = self.dict_vectorized.transform (prepare_row_wise)
        self.Y = np.array (self.Y)
        self.CSC = compressed_row_wise.tocsc ()
    
    def CSC_Inference (self, X):

        CSR = self.dict_vectorized.transform (X)
        return CSR.tocsc ()

    def train_model (self, model):

        model.fit (self.CSC, self.Y)
    
    def RMSE_cross_validation_record (self, model):
        
        predictions = model.predict (self.CSC)
        sum_square = 0
        
        for index in range (len (predictions)):
            prediction = predictions[index]
            observation = self.Y[index]
            sum_square += (prediction - observation) ** 2
        
        RMSE = np.sqrt (sum_square / len (predictions))
        RMSE_Factorization_Machine.append (RMSE)

def processing_Movies ():

    extract_title_Movies = {}
    extract_genre_Movies = {}
    for index in Movies.index:
        movie_ID_Movies = np.int64 (Movies.loc[index]["movieId"])
        title_Movies = Movies.loc[index]["title"]
        genres_Movies = Movies.loc[index]["genres"]

        extract_title_Movies[movie_ID_Movies] = title_Movies
        extract_genre_Movies[movie_ID_Movies] = re.split (r'\|', genres_Movies)
    
    return extract_title_Movies, extract_genre_Movies

def processing_Tags ():

    user_ID_Tags = np.array (Tags["userId"].unique ())
    movie_ID_Tags = np.array (Tags["movieId"].unique ())
    n_user_ID_Tags = len (user_ID_Tags)
    m_movie_ID_Tags = len (movie_ID_Tags)
    
    convert_unique_user_ID_to_unique_index_Tags = {}
    convert_unique_movie_ID_to_unique_index_Tags = {}
    for i in range (n_user_ID_Tags):
        index = user_ID_Tags[i]
        convert_unique_user_ID_to_unique_index_Tags[index] = i
    for i in range (m_movie_ID_Tags):
        index = movie_ID_Tags[i]
        convert_unique_movie_ID_to_unique_index_Tags[index] = i

    temp = Matrix_Factorization ()
    temp.user_movie_matrix_creation (True, "tag", 0, 0, False, n_user_ID_Tags, m_movie_ID_Tags, Tags, convert_unique_user_ID_to_unique_index_Tags, convert_unique_movie_ID_to_unique_index_Tags)
    user_movie_tag = temp.get_user_movie_matrix ()

    return user_movie_tag, convert_unique_user_ID_to_unique_index_Tags, convert_unique_movie_ID_to_unique_index_Tags

def processing_Links ():

    extract_imdb_Links = {}
    extract_tmdb_links = {}

    for index in Links.index:
        movie_ID_Links = np.int64 (Links.loc[index]["movieId"])
        imdb_Links = Links.loc[index]["imdbId"]
        tmdb_Links = Links.loc[index]["tmdbId"]

        if not pd.isna (imdb_Links):
            extract_imdb_Links[movie_ID_Links] = int (imdb_Links)
        if not pd.isna (tmdb_Links):
            extract_tmdb_links[movie_ID_Links] = int (tmdb_Links)
    
    return extract_imdb_Links, extract_tmdb_links

def RMSE_of_MF_and_FM (extract_genre_Movies, user_movie_matrix_Tags, convert_unique_user_ID_to_unique_index_Tags, convert_unique_movie_ID_to_unique_index_Tags, extract_imdb_Links, extract_tmdb_links):

    z = 0
    for train_index, test_index in train_test_split_ratings:

        print (f'\n----------------------------K-Fold: {z + 1}----------------------------')
        z += 1

        train_Ratings = Ratings.iloc[train_index].reset_index (drop = True)
        test_Ratings = Ratings.iloc[test_index].reset_index (drop = True)

        train_user_ID_Ratings = np.array (train_Ratings["userId"].unique ())
        train_movie_ID_Ratings = np.array (train_Ratings["movieId"].unique ())
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

        test_user_ID_Ratings = np.array (test_Ratings["userId"].unique ())
        test_movie_ID_Ratings = np.array (test_Ratings["movieId"].unique ())

        valid_user_ID_Ratings = []
        valid_movie_ID_Ratings = [] 
        for temp in test_user_ID_Ratings:
            if temp in train_convert_unique_user_ID_to_unique_index_Ratings:
                valid_user_ID_Ratings.append (temp)
        for temp in test_movie_ID_Ratings:
            if temp in train_convert_unique_movie_ID_to_unique_index_Ratings:
                valid_movie_ID_Ratings.append (temp)

        test_n_user_ID_Ratings = len (valid_user_ID_Ratings)
        test_m_movie_ID_Ratings = len (valid_movie_ID_Ratings)
        test_convert_unique_user_ID_to_unique_index_Ratings = {}
        test_convert_unique_movie_ID_to_unique_index_Ratings = {}
        for i in range (test_n_user_ID_Ratings):
            index = valid_user_ID_Ratings[i]
            test_convert_unique_user_ID_to_unique_index_Ratings[index] = i
        for i in range (test_m_movie_ID_Ratings):
            index = valid_movie_ID_Ratings[i]
            test_convert_unique_movie_ID_to_unique_index_Ratings[index] = i

        model = SVDpp ()
        MF = Matrix_Factorization ()
        MF.user_movie_matrix_creation (False, "rating", train_user_ID_Ratings, train_movie_ID_Ratings, False, train_n_user_ID_Ratings, train_m_movie_ID_Ratings, train_Ratings, train_convert_unique_user_ID_to_unique_index_Ratings, train_convert_unique_movie_ID_to_unique_index_Ratings)
        train_user_movie_Ratings = MF.get_user_movie_matrix ()
        MF.train_model (model, train_user_ID_Ratings, train_movie_ID_Ratings)
        MF.user_movie_matrix_creation (False, "rating", valid_user_ID_Ratings, valid_movie_ID_Ratings, True, test_n_user_ID_Ratings, test_m_movie_ID_Ratings, test_Ratings, test_convert_unique_user_ID_to_unique_index_Ratings, test_convert_unique_movie_ID_to_unique_index_Ratings)
        test_user_movie_Ratings = MF.get_user_movie_matrix ()
        MF.RMSE_cross_validation_record (model, test_user_ID_Ratings, test_movie_ID_Ratings)

        model = FM_Regression ()
        FM = Factorization_Machine ()
        FM.compressed_column_wise (True, train_user_ID_Ratings, train_movie_ID_Ratings, train_user_movie_Ratings, user_movie_matrix_Tags, extract_genre_Movies, convert_unique_user_ID_to_unique_index_Tags, convert_unique_movie_ID_to_unique_index_Tags)
        FM.train_model (model)
        FM.compressed_column_wise (False, valid_user_ID_Ratings, valid_movie_ID_Ratings, test_user_movie_Ratings, user_movie_matrix_Tags, extract_genre_Movies, convert_unique_user_ID_to_unique_index_Tags, convert_unique_movie_ID_to_unique_index_Tags)
        FM.RMSE_cross_validation_record (model)

def RMSE_GRAPH_of_MF_and_FM ():
    
    plt.figure (figsize = (8, 6))
    graph = sns.boxplot (data = [RMSE_Matrix_Factorization, RMSE_Factorization_Machine])

    plt.title ("RMSE COMPARISON of MATRIX FACTORIZATION & FACTORIZATION MACHINE")
    graph.set_ylabel ("Root Mean Square Error")
    graph.set_xticks ([0, 1])
    graph.set_xticklabels (["MATRIX FACTORIZATION", "FACTORIZATION MACHINE"])

    plt.savefig ("RMSE_COMPARISON.png")
    plt.show ()

if __name__ == "__main__":

    extract_title_Movies, extract_genre_Movies = processing_Movies ()
    user_movie_matrix_Tags, convert_unique_user_ID_to_unique_index_Tags, convert_unique_movie_ID_to_unique_index_Tags = processing_Tags ()
    extract_imdb_Links, extract_tmdb_links = processing_Links ()
    RMSE_of_MF_and_FM (extract_genre_Movies, user_movie_matrix_Tags, convert_unique_user_ID_to_unique_index_Tags, convert_unique_movie_ID_to_unique_index_Tags, extract_imdb_Links, extract_tmdb_links)
    RMSE_GRAPH_of_MF_and_FM () 

    joblib.dump (user_movie_matrix_Tags, "user_movie_matrix_Tags.pkl")
    joblib.dump (extract_genre_Movies, "extract_genre_Movies.pkl")
    joblib.dump (convert_unique_user_ID_to_unique_index_Tags, "convert_unique_user_ID_to_unique_index_Tags.pkl")
    joblib.dump (convert_unique_movie_ID_to_unique_index_Tags, "convert_unique_movie_ID_to_unique_index_Tags.pkl")

    unique_movie_id = np.array (Movies["movieId"].unique ())
    unique_movie_genre = {}
    for key, value in extract_genre_Movies.items ():
        for string in value:
            unique_movie_genre[string] = True

    joblib.dump (unique_movie_id, "unique_movie_id.pkl")
    joblib.dump (unique_movie_genre, "unique_movie_genre.pkl")
    joblib.dump (extract_title_Movies, "extract_title_Movies.pkl")
    joblib.dump (extract_imdb_Links, "extract_imdb_Links.pkl")
    joblib.dump (extract_tmdb_links, "extract_tmdb_links.pkl")


# cd "/Users/chikhoado/Desktop/PROJECTS/Movie Recommendation"
# /opt/homebrew/bin/python3.12 -m venv .venv
# source .venv/bin/activate
# pip install pandas scikit-learn numpy seaborn joblib scikit-surprise "numpy<2.0.0" 
# python "/Users/chikhoado/Desktop/PROJECTS/Movie Recommendation/Training_Phase_1.py"





