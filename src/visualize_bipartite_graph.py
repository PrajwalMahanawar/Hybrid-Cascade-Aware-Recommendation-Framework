import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

interactions = pd.read_csv(
    "/Users/prajwalmahanawar/cascade_recommender/data/interactions.csv"
)

# Use subset for readable visualization
sample = interactions.head(500)

G = nx.Graph()

users = set()
movies = set()

for _, row in sample.iterrows():
    user = f"U{row['user_id']}"
    movie = f"M{row['movie_id']}"

    G.add_node(user, node_type="user")
    G.add_node(movie, node_type="movie")

    G.add_edge(user, movie, weight=row["rating"])

    users.add(user)
    movies.add(movie)

pos = nx.spring_layout(G, seed=42, k=0.5)

plt.figure(figsize=(14, 10))

nx.draw_networkx_nodes(G, pos, nodelist=list(users), node_size=40, label="Users")

nx.draw_networkx_nodes(
    G, pos, nodelist=list(movies), node_size=80, node_shape="s", label="Movies"
)

nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)

plt.title("User-Movie Bipartite Interaction Graph")
plt.legend()
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "/Users/prajwalmahanawar/cascade_recommender/outputs/user_movie_bipartite_graph.png",
    dpi=300,
)

plt.show()

print("Saved: outputs/user_movie_bipartite_graph.png")
