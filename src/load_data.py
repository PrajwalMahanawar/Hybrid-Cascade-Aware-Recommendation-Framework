import pandas as pd

users = pd.read_csv("/Users/prajwalmahanawar/cascade_recommender/data/users.csv")

movies = pd.read_csv("/Users/prajwalmahanawar/cascade_recommender/data/movies.csv")

interactions = pd.read_csv(
    "/Users/prajwalmahanawar/cascade_recommender/data/interactions.csv"
)

print("Users:")
print(users.head())

print("\nMovies:")
print(movies.head())

print("\nInteractions:")
print(interactions.head())
