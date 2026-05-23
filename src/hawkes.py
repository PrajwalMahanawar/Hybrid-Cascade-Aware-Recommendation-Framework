import math
import pandas as pd


def temporal_intensity(movie_node, interactions, decay=0.1):

    interactions = interactions.copy()

    interactions.loc[:, "timestamp"] = pd.to_datetime(interactions["timestamp"])

    # Convert M123 -> 123
    movie_id = int(str(movie_node).replace("M", ""))

    movie_events = interactions[interactions["movie_id"] == movie_id]

    if movie_events.empty:
        return 0

    latest_time = interactions["timestamp"].max()

    intensity = 0

    for _, row in movie_events.iterrows():
        time_diff_hours = (latest_time - row["timestamp"]).total_seconds() / 3600

        intensity += math.exp(-decay * time_diff_hours)

    return intensity
