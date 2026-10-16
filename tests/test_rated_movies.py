from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rated_movies import load_rated_movies


class RatedMoviesTests(unittest.TestCase):
    def test_utf8_bom_header_and_repeated_ratings(self):
        with TemporaryDirectory() as directory:
            ratings_path = Path(directory) / "ratings.csv"
            ratings_path.write_text(
                "userId,movieId,rating,timestamp\n"
                "7,10,4.0,1\n"
                "7,10,5.0,2\n"
                "7,20,3.0,3\n",
                encoding="utf-8-sig",
            )

            self.assertEqual(load_rated_movies(ratings_path)[7], {10, 20})


if __name__ == "__main__":
    unittest.main()
