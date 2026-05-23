import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from icm import independent_cascade
from ltm import linear_threshold
from hawkes import temporal_intensity


def min_max_normalize(value, min_value, max_value):
    if max_value == min_value:
        return 0
    return (value - min_value) / (max_value - min_value)


interactions = pd.read_csv(
    "/Users/prajwalmahanawar/cascade_recommender/data/interactions.csv"
)

interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])

# Use subset for speed
interactions = interactions.head(50000)

G = nx.Graph()

for _, row in interactions.iterrows():
    user = f"U{row['user_id']}"
    movie = f"M{row['movie_id']}"

    G.add_edge(user, movie, weight=row["rating"])

movie_nodes = [node for node in G.nodes() if str(node).startswith("M")]

# Analyze top 10 popular movies
movie_nodes = sorted(movie_nodes, key=lambda node: G.degree(node), reverse=True)[:10]

raw_scores = {}

for movie in movie_nodes:
    icm_score, _ = independent_cascade(G, movie, probability=0.3, steps=4)

    ltm_score, _ = linear_threshold(G, movie, threshold=0.5, steps=4)

    hawkes_score = temporal_intensity(movie, interactions)

    raw_scores[movie] = {"ICM": icm_score, "LTM": ltm_score, "Hawkes": hawkes_score}

icm_values = [v["ICM"] for v in raw_scores.values()]
ltm_values = [v["LTM"] for v in raw_scores.values()]
hawkes_values = [v["Hawkes"] for v in raw_scores.values()]

icm_min, icm_max = min(icm_values), max(icm_values)
ltm_min, ltm_max = min(ltm_values), max(ltm_values)
hawkes_min, hawkes_max = min(hawkes_values), max(hawkes_values)

rows = []

for movie, scores in raw_scores.items():
    icm_norm = min_max_normalize(scores["ICM"], icm_min, icm_max)
    ltm_norm = min_max_normalize(scores["LTM"], ltm_min, ltm_max)
    hawkes_norm = min_max_normalize(scores["Hawkes"], hawkes_min, hawkes_max)

    rows.append(
        {"movie": movie, "ICM": icm_norm, "LTM": ltm_norm, "Hawkes": hawkes_norm}
    )

df = pd.DataFrame(rows)

df.to_csv(
    "/Users/prajwalmahanawar/cascade_recommender/outputs/hybrid_component_scores.csv",
    index=False,
)

plt.figure(figsize=(12, 6))

bottom_ltm = df["ICM"]
bottom_hawkes = df["ICM"] + df["LTM"]

plt.bar(df["movie"], df["ICM"], label="ICM")

plt.bar(df["movie"], df["LTM"], bottom=bottom_ltm, label="LTM")

plt.bar(df["movie"], df["Hawkes"], bottom=bottom_hawkes, label="Hawkes")

plt.title("Normalized Hybrid Score Component Analysis")
plt.xlabel("Movie")
plt.ylabel("Normalized Component Contribution")

plt.legend()
plt.tight_layout()

plt.savefig(
    "/Users/prajwalmahanawar/cascade_recommender/outputs/hybrid_component_analysis.png",
    dpi=300,
)

plt.show()

print("Saved: outputs/hybrid_component_analysis.png")
print("Saved: outputs/hybrid_component_scores.csv")
