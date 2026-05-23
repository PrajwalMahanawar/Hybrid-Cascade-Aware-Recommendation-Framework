import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from icm import independent_cascade

interactions = pd.read_csv(
    "/Users/prajwalmahanawar/cascade_recommender/data/interactions.csv"
)

# Use subset for readable graph
sample = interactions.head(10000)

G = nx.Graph()

for _, row in sample.iterrows():
    user = f"U{row['user_id']}"
    movie = f"M{row['movie_id']}"

    G.add_edge(user, movie, weight=row["rating"])

# Choose a movie with many connections
movie_nodes = [node for node in G.nodes() if str(node).startswith("M")]

seed_movie = max(movie_nodes, key=lambda node: G.degree(node))

spread_size, activated_nodes = independent_cascade(
    G, seed_movie, probability=0.3, steps=4
)

cascade_subgraph = G.subgraph(activated_nodes)

pos = nx.spring_layout(cascade_subgraph, seed=42, k=0.8)

plt.figure(figsize=(12, 8))

user_nodes = [node for node in cascade_subgraph.nodes() if str(node).startswith("U")]

movie_nodes = [node for node in cascade_subgraph.nodes() if str(node).startswith("M")]

nx.draw_networkx_nodes(
    cascade_subgraph, pos, nodelist=user_nodes, node_size=250, label="Activated Users"
)

nx.draw_networkx_nodes(
    cascade_subgraph,
    pos,
    nodelist=movie_nodes,
    node_size=500,
    node_shape="s",
    label="Activated Movies",
)

nx.draw_networkx_edges(cascade_subgraph, pos, alpha=0.4, width=1.2)

nx.draw_networkx_labels(cascade_subgraph, pos, font_size=8)

plt.title(
    f"ICM Cascade Propagation from Seed Movie {seed_movie}\nActivated Nodes: {spread_size}"
)

plt.legend()
plt.axis("off")
plt.tight_layout()

plt.savefig(
    "/Users/prajwalmahanawar/cascade_recommender/outputs/icm_cascade_propagation.png",
    dpi=300,
)

plt.show()

print(f"Seed Movie: {seed_movie}")
print(f"Activated Nodes: {activated_nodes}")
print("Saved: outputs/icm_cascade_propagation.png")
