import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from icm import independent_cascade
from ltm import linear_threshold

interactions = pd.read_csv(
    "/Users/prajwalmahanawar/cascade_recommender/data/interactions.csv"
)

# Use subset for speed
interactions = interactions.head(50000)

G = nx.Graph()

for _, row in interactions.iterrows():
    user = f"U{row['user_id']}"
    movie = f"M{row['movie_id']}"

    G.add_edge(user, movie, weight=row["rating"])

movie_nodes = [node for node in G.nodes() if str(node).startswith("M")]

# Use top 100 movies for speed
movie_nodes = sorted(movie_nodes, key=lambda node: G.degree(node), reverse=True)[:100]

icm_spreads = []
ltm_spreads = []

for movie in movie_nodes:
    icm_size, _ = independent_cascade(G, movie, probability=0.3, steps=4)

    ltm_size, _ = linear_threshold(G, movie, threshold=0.5, steps=4)

    icm_spreads.append(icm_size)
    ltm_spreads.append(ltm_size)

plt.figure(figsize=(10, 6))

plt.hist(icm_spreads, bins=20, alpha=0.7, label="ICM Spread")

plt.hist(ltm_spreads, bins=20, alpha=0.7, label="LTM Spread")

plt.title("Diffusion Spread Distribution")
plt.xlabel("Activated Nodes / Spread Size")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()

plt.savefig(
    "/Users/prajwalmahanawar/cascade_recommender/outputs/diffusion_spread_distribution.png",
    dpi=300,
)

plt.show()

print("Saved: outputs/diffusion_spread_distribution.png")
print("Average ICM Spread:", sum(icm_spreads) / len(icm_spreads))
print("Average LTM Spread:", sum(ltm_spreads) / len(ltm_spreads))
