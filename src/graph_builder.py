import pandas as pd
import networkx as nx

interactions = pd.read_csv(
    "/Users/prajwalmahanawar/cascade_recommender/data/interactions.csv"
)

G = nx.Graph()

# Add user-movie edges
for _, row in interactions.iterrows():
    user = f"U{row['user_id']}"
    movie = f"M{row['movie_id']}"

    rating = row["rating"]

    G.add_edge(user, movie, weight=rating)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
