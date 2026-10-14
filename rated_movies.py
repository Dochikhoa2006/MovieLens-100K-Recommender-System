import csv
from collections import defaultdict
from pathlib import Path


DEFAULT_RATINGS_PATH = Path(__file__).resolve().parent / "MovieLens_100k_Dataset/ratings.csv"


def load_rated_movies(path=DEFAULT_RATINGS_PATH):
    """Return movie IDs each user has already rated."""
    rated_movies = defaultdict(set)
    with open(path, newline="", encoding="utf-8") as ratings_file:
        for row in csv.DictReader(ratings_file):
            rated_movies[int(row["userId"])].add(int(row["movieId"]))
    return rated_movies
