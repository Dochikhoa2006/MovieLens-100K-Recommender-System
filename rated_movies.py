import csv
from collections import defaultdict


def load_rated_movies(path="MovieLens_100k_Dataset/ratings.csv"):
    """Return movie IDs each user has already rated."""
    rated_movies = defaultdict(set)
    with open(path, newline="", encoding="utf-8") as ratings_file:
        for row in csv.DictReader(ratings_file):
            rated_movies[int(row["userId"])].add(int(row["movieId"]))
    return rated_movies
