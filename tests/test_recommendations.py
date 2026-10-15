import unittest

from Inference import parse_genre_choices
from recommendation_candidates import build_candidates


class RecommendationTests(unittest.TestCase):
    def test_candidates_use_movie_genres_and_skip_rated_movies(self):
        features, movie_ids = build_candidates(
            user_id=7,
            selected_genres=["Adventure"],
            movie_ids=[10, 11, 12, 13],
            genres_by_movie={
                10: ["Adventure", "Comedy"],
                11: ["Adventure"],
                12: ["Comedy"],
                13: [],
            },
            rated_movie_ids={11},
        )

        self.assertEqual(movie_ids, [10])
        self.assertEqual(
            features,
            [{"USER_ID": 7, "MOVIE_ID": 10, "Adventure": "Adventure", "Comedy": "Comedy"}],
        )

    def test_genre_names_numbers_aliases_and_invalid_entries(self):
        selected, invalid = parse_genre_choices(
            " adventure,2,Adventure,none of the above,99, ",
            ["Adventure", "Action", "(no genres listed)"],
        )

        self.assertEqual(selected, ["Adventure", "Action", "(no genres listed)"])
        self.assertEqual(invalid, ["99", "(blank)"])


if __name__ == "__main__":
    unittest.main()
