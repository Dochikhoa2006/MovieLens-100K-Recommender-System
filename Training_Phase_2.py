from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from Factorization_Machine_Library import FM_Regression
from Training_Phase_1 import Factorization_Machine, Matrix_Factorization


BASE_DIR = Path(__file__).resolve().parent


def train_final_model(base_dir=BASE_DIR):
    """Train the final FM model and save its inference artifacts."""
    base_dir = Path(base_dir)
    ratings = pd.read_csv(base_dir / "MovieLens_100k_Dataset/ratings.csv")
    user_movie_tags = joblib.load(base_dir / "user_movie_matrix_Tags.pkl")
    movie_genres = joblib.load(base_dir / "extract_genre_Movies.pkl")
    tag_user_indices = joblib.load(base_dir / "convert_unique_user_ID_to_unique_index_Tags.pkl")
    tag_movie_indices = joblib.load(base_dir / "convert_unique_movie_ID_to_unique_index_Tags.pkl")

    user_ids = np.array(ratings["userId"].unique())
    movie_ids = np.array(ratings["movieId"].unique())
    user_indices = {user_id: index for index, user_id in enumerate(user_ids)}
    movie_indices = {movie_id: index for index, movie_id in enumerate(movie_ids)}

    matrix = Matrix_Factorization()
    matrix.user_movie_matrix_creation(
        False, "rating", user_ids, movie_ids, False, len(user_ids), len(movie_ids),
        ratings, user_indices, movie_indices,
    )

    model = FM_Regression(random_state=150)
    fm = Factorization_Machine()
    fm.compressed_column_wise(
        True, user_ids, movie_ids, matrix.get_user_movie_matrix(), user_movie_tags,
        movie_genres, tag_user_indices, tag_movie_indices,
    )
    fm.train_model(model)

    joblib.dump(model, base_dir / "Factorization_Model_Trained.pkl")
    joblib.dump(fm, base_dir / "FM.pkl")
    return model, fm


if __name__ == "__main__":
    train_final_model()
