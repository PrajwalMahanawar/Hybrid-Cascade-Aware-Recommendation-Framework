import networkx as nx

from icm import independent_cascade
from ltm import linear_threshold
from hawkes import temporal_intensity


def min_max_normalize(value, min_value, max_value):
    if max_value == min_value:
        return 0
    return (value - min_value) / (max_value - min_value)


def build_graph(interactions):
    G = nx.Graph()

    for _, row in interactions.iterrows():
        user = f"U{row['user_id']}"
        movie = f"M{row['movie_id']}"
        rating = row["rating"]

        G.add_edge(user, movie, weight=rating)

    return G


def normalized_hybrid_recommend(user_id, interactions, k=3):
    G = build_graph(interactions)

    if user_id not in G:
        return []

    interacted = set(G.neighbors(user_id))
    raw_scores = {}

    for node in G.nodes():
        if str(node).startswith("M") and node not in interacted:
            icm_score, _ = independent_cascade(G, node, probability=0.3)

            ltm_score, _ = linear_threshold(G, node, threshold=0.5)

            hawkes_score = temporal_intensity(node, interactions)

            raw_scores[node] = {
                "icm": icm_score,
                "ltm": ltm_score,
                "hawkes": hawkes_score,
            }

    if not raw_scores:
        return []

    icm_values = [v["icm"] for v in raw_scores.values()]
    ltm_values = [v["ltm"] for v in raw_scores.values()]
    hawkes_values = [v["hawkes"] for v in raw_scores.values()]

    icm_min, icm_max = min(icm_values), max(icm_values)
    ltm_min, ltm_max = min(ltm_values), max(ltm_values)
    hawkes_min, hawkes_max = min(hawkes_values), max(hawkes_values)

    final_scores = {}

    for node, scores in raw_scores.items():
        icm_norm = min_max_normalize(scores["icm"], icm_min, icm_max)

        ltm_norm = min_max_normalize(scores["ltm"], ltm_min, ltm_max)

        hawkes_norm = min_max_normalize(scores["hawkes"], hawkes_min, hawkes_max)

        hybrid_score = 0.4 * icm_norm + 0.2 * ltm_norm + 0.4 * hawkes_norm

        final_scores[node] = hybrid_score

    return [
        item
        for item, score in sorted(
            final_scores.items(), key=lambda x: x[1], reverse=True
        )[:k]
    ]
