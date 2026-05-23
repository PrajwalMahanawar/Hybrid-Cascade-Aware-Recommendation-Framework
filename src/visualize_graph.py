import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

interactions = pd.read_csv(
    "/Users/prajwalmahanawar/cascade_recommender/data/interactions.csv"
)

G = nx.Graph()

# Use smaller sample for visualization
sample_interactions = interactions.sample(n=200, random_state=42)

for _, row in sample_interactions.iterrows():
    user = f"U{row['user_id']}"
    movie = f"M{row['movie_id']}"

    G.add_edge(user, movie, weight=row["rating"])

plt.figure(figsize=(14, 10))

pos = nx.spring_layout(G, seed=42)

nx.draw(G, pos, with_labels=False, node_size=40, alpha=0.7)

plt.title("MovieLens User-Movie Interaction Network")

plt.savefig(
    "/Users/prajwalmahanawar/cascade_recommender/outputs/movie_interaction_network.png"
)

plt.show()

print("\nGraph visualization saved successfully.")
