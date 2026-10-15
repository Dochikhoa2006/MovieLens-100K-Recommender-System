import csv
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from Inference import print_recommended_movie
from recommendation_export import recommendations_csv


class RecommendationExportTests(unittest.TestCase):
    def test_export_preserves_rank_metadata_and_escapes_csv_text(self):
        exported = recommendations_csv(
            recommendations=[(4.25, 10), (3.5, 20)],
            titles={10: 'Movie, the "Best"', 20: "=1+1"},
            genres_by_movie={10: ["Adventure", "Comedy"], 20: ["(no genres listed)"]},
            imdb_links={10: 12345},
            tmdb_links={10: 99},
        )
        rows = list(csv.DictReader(StringIO(exported)))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["rank"], "1")
        self.assertEqual(rows[0]["title"], 'Movie, the "Best"')
        self.assertEqual(rows[0]["predictedRating"], "4.25")
        self.assertEqual(rows[0]["genres"], "Adventure, Comedy")
        self.assertEqual(rows[0]["imdbUrl"], "https://www.imdb.com/title/tt0012345/")
        self.assertEqual(rows[0]["tmdbUrl"], "https://www.themoviedb.org/movie/99/")
        self.assertEqual(rows[1]["title"], "'=1+1")
        self.assertEqual(rows[1]["genres"], "No genres listed")
        self.assertEqual(rows[1]["imdbUrl"], "")

    def test_cli_can_save_the_same_recommendation_as_csv(self):
        class Encoder:
            def CSC_Inference(self, features):
                return features

        class Model:
            def predict(self, features):
                return [4.2]

        assets = (
            Model(), Encoder(), [10], {"Action": True}, {10: ["Action"]},
            {10: "Test Movie"}, {10: 12345}, {10: 99}, {7: set()},
        )
        with TemporaryDirectory() as directory:
            output = Path(directory) / "recommendations.csv"
            with patch("Inference.load_assets", return_value=assets):
                with redirect_stdout(StringIO()):
                    print_recommended_movie(1, user_id=7, genre_choices="Action", output_path=output)

            with output.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["title"], "Test Movie")
        self.assertEqual(rows[0]["predictedRating"], "4.20")


if __name__ == "__main__":
    unittest.main()
