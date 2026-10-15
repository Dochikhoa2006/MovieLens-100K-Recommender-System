import unittest

from Inference import parse_genre_choices
from recommendation_candidates import build_candidates, rank_recommendations


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

    def test_ranking_handles_ties_and_top_k(self):
        class Encoder:
            def CSC_Inference(self, features):
                return features

        class Model:
            def predict(self, features):
                return [4.0 if row["MOVIE_ID"] in (20, 30) else 3.0 for row in features]

        results = rank_recommendations(
            Model(), Encoder(), 7, ["Action"], [10, 20, 30],
            {10: ["Action"], 20: ["Action"], 30: ["Action"]}, {10}, 1,
        )
        self.assertEqual(results, [(4.0, 30)])

    def test_empty_candidates_skip_model_prediction(self):
        class Unused:
            def __getattr__(self, name):
                raise AssertionError(f"{name} should not be called")

        results = rank_recommendations(
            Unused(), Unused(), 7, ["Comedy"], [10], {10: ["Action"]}, set(), 5,
        )
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
