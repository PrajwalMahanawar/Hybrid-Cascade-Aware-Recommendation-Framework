import pandas as pd
import networkx as nx

from icm import independent_cascade
from ltm import linear_threshold
from hawkes import temporal_intensity


# -----------------------------
# Load data
# -----------------------------

interactions = pd.read_csv(
    "/Users/prajwalmahanawar/cascade_recommender/data/interactions.csv"
)

movies = pd.read_csv("/Users/prajwalmahanawar/cascade_recommender/data/movies.csv")

interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])

# Use manageable subset
interactions = interactions.head(50000)

# -----------------------------
# Helper functions
# -----------------------------


def movie_title(movie_node):
    movie_id = int(str(movie_node).replace("M", ""))

    row = movies[movies["movie_id"] == movie_id]

    if row.empty:
        return movie_node

    return row.iloc[0]["title"]


def build_graph(data):
    G = nx.Graph()

    for _, row in data.iterrows():
        user = f"U{row['user_id']}"
        movie = f"M{row['movie_id']}"

        G.add_edge(user, movie, weight=row["rating"])

    return G


def min_max_normalize(value, min_value, max_value):
    if max_value == min_value:
        return 0

    return (value - min_value) / (max_value - min_value)


# -----------------------------
# Build graph
# -----------------------------

G = build_graph(interactions)


# -----------------------------
# Explain recommendation
# -----------------------------


def explain_recommendations(user_node, k=5):

    if user_node not in G:
        print(f"{user_node} not found in graph.")
        return

    watched_movies = [
        node for node in G.neighbors(user_node) if str(node).startswith("M")
    ]

    raw_scores = {}

    for node in G.nodes():
        if str(node).startswith("M") and node not in watched_movies:
            icm_score, _ = independent_cascade(G, node, probability=0.3)

            ltm_score, _ = linear_threshold(G, node, threshold=0.5)

            hawkes_score = temporal_intensity(node, interactions)

            raw_scores[node] = {
                "icm": icm_score,
                "ltm": ltm_score,
                "hawkes": hawkes_score,
            }

    if not raw_scores:
        print(f"No recommendations available for {user_node}.")
        return

    icm_values = [v["icm"] for v in raw_scores.values()]
    ltm_values = [v["ltm"] for v in raw_scores.values()]
    hawkes_values = [v["hawkes"] for v in raw_scores.values()]

    icm_min, icm_max = min(icm_values), max(icm_values)
    ltm_min, ltm_max = min(ltm_values), max(ltm_values)
    hawkes_min, hawkes_max = min(hawkes_values), max(hawkes_values)

    final_scores = {}

    for movie, scores in raw_scores.items():
        icm_norm = min_max_normalize(scores["icm"], icm_min, icm_max)

        ltm_norm = min_max_normalize(scores["ltm"], ltm_min, ltm_max)

        hawkes_norm = min_max_normalize(scores["hawkes"], hawkes_min, hawkes_max)

        hybrid_score = 0.4 * icm_norm + 0.2 * ltm_norm + 0.4 * hawkes_norm

        final_scores[movie] = {
            "hybrid_score": hybrid_score,
            "icm_norm": icm_norm,
            "ltm_norm": ltm_norm,
            "hawkes_norm": hawkes_norm,
        }

    recommendations = sorted(
        final_scores.items(), key=lambda x: x[1]["hybrid_score"], reverse=True
    )[:k]

    print("\n================================================")
    print(f"User: {user_node}")
    print("================================================")

    print("\nPreviously Watched Movies:")
    for movie in watched_movies[:10]:
        print(f"- {movie_title(movie)}")

    print("\nTop Recommendations:")
    for rank, (movie, scores) in enumerate(recommendations, start=1):
        print(f"\n{rank}. {movie_title(movie)}")
        print(f"   Movie ID: {movie}")
        print(f"   Hybrid Score: {scores['hybrid_score']:.4f}")
        print(f"   ICM Component: {scores['icm_norm']:.4f}")
        print(f"   LTM Component: {scores['ltm_norm']:.4f}")
        print(f"   Hawkes Component: {scores['hawkes_norm']:.4f}")

    print("\nInterpretation Guide:")
    print("- High ICM means strong cascade spread potential.")
    print("- High LTM means threshold-based activation support.")
    print("- High Hawkes means strong temporal/recent interaction intensity.")


# -----------------------------
# Run examples
# -----------------------------

example_users = ["U1", "U10", "U50"]

for user in example_users:
    explain_recommendations(user, k=5)
