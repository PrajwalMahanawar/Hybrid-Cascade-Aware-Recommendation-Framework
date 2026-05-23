import pandas as pd
import networkx as nx

interactions = pd.read_csv(
    "/Users/prajwalmahanawar/cascade_recommender/data/interactions.csv"
)

G = nx.Graph()

# Build user-movie graph
for _, row in interactions.iterrows():
    user = f"U{row['user_id']}"
    movie = f"M{row['movie_id']}"

    rating = row["rating"]

    G.add_edge(user, movie, weight=rating)


def recommend_movies(user_id):

    interacted = set(G.neighbors(user_id))

    scores = {}

    for node in G.nodes():
        # Only movie nodes
        if str(node).startswith("M") and node not in interacted:
            scores[node] = G.degree(node)

    recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return recommendations[:10]


print(recommend_movies("U1"))
