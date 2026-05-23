import pandas as pd
import networkx as nx
from hawkes import temporal_intensity

interactions = pd.read_csv(
    "/Users/prajwalmahanawar/cascade_recommender/data/interactions.csv"
)

interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])

G = nx.Graph()

# Build user-movie graph
for _, row in interactions.iterrows():
    user = f"U{row['user_id']}"
    movie = f"M{row['movie_id']}"
    rating = row["rating"]

    G.add_edge(user, movie, weight=rating)


def recommend_with_hawkes(user_id):
    if user_id not in G:
        return []

    interacted = set(G.neighbors(user_id))
    scores = {}

    for node in G.nodes():
        if str(node).startswith("M") and node not in interacted:
            score = temporal_intensity(node, interactions)

            scores[node] = score

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]


print(recommend_with_hawkes("U1"))
