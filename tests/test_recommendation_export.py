import csv
from io import StringIO
import unittest

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


if __name__ == "__main__":
    unittest.main()
