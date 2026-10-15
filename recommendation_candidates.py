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


def rank_recommendations(
    model, feature_encoder, user_id, selected_genres, movie_ids,
    genres_by_movie, rated_movie_ids, top_k,
):
    """Score matching unseen movies and return the highest-ranked results."""
    features, candidate_ids = build_candidates(
        user_id, selected_genres, movie_ids, genres_by_movie, rated_movie_ids
    )
    if not features:
        return []

    scores = model.predict(feature_encoder.CSC_Inference(features))
    return sorted(zip(scores, candidate_ids), reverse=True)[:top_k]


def format_movie_genres(genres):
    """Format movie metadata for recommendation details."""
    if not genres:
        return "No genres listed"
    return ", ".join(
        "No genres listed" if genre == "(no genres listed)" else genre
        for genre in genres
    )
