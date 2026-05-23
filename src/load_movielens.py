import pandas as pd

ratings = pd.read_csv(
    "/Users/prajwalmahanawar/cascade_recommender/data/ratings.dat",
    sep="::",
    engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"],
)

movies = pd.read_csv(
    "/Users/prajwalmahanawar/cascade_recommender/data/movies.dat",
    sep="::",
    engine="python",
    encoding="latin1",
    names=["movie_id", "title", "genres"],
)

users = pd.read_csv(
    "/Users/prajwalmahanawar/cascade_recommender/data/users.dat",
    sep="::",
    engine="python",
    names=["user_id", "gender", "age", "occupation", "zip"],
)

ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")

ratings.to_csv(
    "/Users/prajwalmahanawar/cascade_recommender/data/interactions.csv", index=False
)
movies.to_csv(
    "/Users/prajwalmahanawar/cascade_recommender/data/movies.csv", index=False
)
users.to_csv("/Users/prajwalmahanawar/cascade_recommender/data/users.csv", index=False)

print("MovieLens converted successfully")
print(ratings.head())
