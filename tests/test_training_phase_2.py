from contextlib import redirect_stdout
from io import StringIO
import importlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import Training_Phase_2


class FinalTrainingTests(unittest.TestCase):
    def test_import_does_not_read_data_or_train(self):
        with patch("pandas.read_csv", side_effect=AssertionError("CSV read during import")):
            with patch("joblib.load", side_effect=AssertionError("artifact loaded during import")):
                importlib.reload(Training_Phase_2)

    def test_tiny_training_run_writes_usable_artifacts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "MovieLens_100k_Dataset"
            dataset.mkdir()
            pd.DataFrame(
                {
                    "userId": [1, 1, 2],
                    "movieId": [1, 2, 1],
                    "rating": [4.0, 3.0, 5.0],
                    "timestamp": [1, 2, 3],
                }
            ).to_csv(dataset / "ratings.csv", index=False)
            joblib.dump(np.empty((0, 0), dtype=object), root / "user_movie_matrix_Tags.pkl")
            joblib.dump({1: ["Action"], 2: ["Comedy"]}, root / "extract_genre_Movies.pkl")
            joblib.dump({}, root / "convert_unique_user_ID_to_unique_index_Tags.pkl")
            joblib.dump({}, root / "convert_unique_movie_ID_to_unique_index_Tags.pkl")

            with redirect_stdout(StringIO()):
                model, fm = Training_Phase_2.train_final_model(root)

            self.assertTrue((root / "Factorization_Model_Trained.pkl").is_file())
            self.assertTrue((root / "FM.pkl").is_file())
            prediction = model.predict(
                fm.CSC_Inference([{"USER_ID": 1, "MOVIE_ID": 1, "Action": "Action"}])
            )
            self.assertEqual(prediction.shape, (1,))


if __name__ == "__main__":
    unittest.main()
