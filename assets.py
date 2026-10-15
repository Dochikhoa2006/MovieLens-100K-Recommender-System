REQUIRED_INFERENCE_FILES = (
    "Factorization_Model_Trained.pkl",
    "FM.pkl",
    "unique_movie_id.pkl",
    "unique_movie_genre.pkl",
    "extract_genre_Movies.pkl",
    "extract_title_Movies.pkl",
    "extract_imdb_Links.pkl",
    "extract_tmdb_links.pkl",
    "MovieLens_100k_Dataset/ratings.csv",
)


def require_inference_assets(base_dir):
    missing = [name for name in REQUIRED_INFERENCE_FILES if not (base_dir / name).is_file()]
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing recommendation files: {names}. See the Set Up section in README.md."
        )
