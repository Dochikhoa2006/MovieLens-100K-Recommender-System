import csv
from io import StringIO

from recommendation_candidates import format_movie_genres


def _safe_csv_text(value):
    text = str(value)
    if text.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


def recommendations_csv(recommendations, titles, genres_by_movie, imdb_links, tmdb_links):
    """Export ranked recommendations with movie metadata as CSV text."""
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["rank", "movieId", "title", "predictedRating", "genres", "imdbUrl", "tmdbUrl"])

    for rank, (score, movie_id) in enumerate(recommendations, start=1):
        imdb_id = imdb_links.get(movie_id)
        tmdb_id = tmdb_links.get(movie_id)
        writer.writerow([
            rank,
            movie_id,
            _safe_csv_text(titles.get(movie_id, "Unknown Title")),
            f"{score:.2f}",
            _safe_csv_text(format_movie_genres(genres_by_movie.get(movie_id, []))),
            f"https://www.imdb.com/title/tt{str(imdb_id).zfill(7)}/" if imdb_id is not None else "",
            f"https://www.themoviedb.org/movie/{tmdb_id}/" if tmdb_id is not None else "",
        ])

    return output.getvalue()
