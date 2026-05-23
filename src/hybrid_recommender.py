import networkx as nx

from icm import independent_cascade
from ltm import linear_threshold
from hawkes import temporal_intensity


def build_graph(interactions):
    G = nx.Graph()

    for _, row in interactions.iterrows():
        user = f"U{row['user_id']}"
        movie = f"M{row['movie_id']}"
        rating = row["rating"]

        G.add_edge(user, movie, weight=rating)

    return G


def hybrid_recommend(user_id, interactions, k=10):
    G = build_graph(interactions)

    if user_id not in G:
        return []

    interacted = set(G.neighbors(user_id))
    scores = {}

    for node in G.nodes():
        if str(node).startswith("M") and node not in interacted:
            icm_score, _ = independent_cascade(G, node, probability=0.3)

            ltm_score, _ = linear_threshold(G, node, threshold=0.5)

            hawkes_score = temporal_intensity(node, interactions)

            hybrid_score = 0.4 * icm_score + 0.2 * ltm_score + 0.4 * hawkes_score

            scores[node] = {
                "hybrid_score": hybrid_score,
                "icm_score": icm_score,
                "ltm_score": ltm_score,
                "hawkes_score": hawkes_score,
            }

    return sorted(scores.items(), key=lambda x: x[1]["hybrid_score"], reverse=True)[:k]
