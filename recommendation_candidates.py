def build_candidates(user_id, selected_genres, movie_ids, genres_by_movie, rated_movie_ids):
    """Build model features for unrated movies in the selected genres."""
    selected = set(selected_genres)
    features = []
    candidate_ids = []

    for movie_id in movie_ids:
        if movie_id in rated_movie_ids:
            continue

        movie_genres = genres_by_movie.get(movie_id, ())
        if not selected.intersection(movie_genres):
            continue

        datapoint = {"USER_ID": user_id, "MOVIE_ID": movie_id}
        for genre in movie_genres:
            datapoint[genre] = genre

        features.append(datapoint)
        candidate_ids.append(movie_id)

    return features, candidate_ids
