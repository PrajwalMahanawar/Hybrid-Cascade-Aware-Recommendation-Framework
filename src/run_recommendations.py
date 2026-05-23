import pandas as pd

from hybrid_recommender import hybrid_recommend

users = pd.read_csv("/Users/prajwalmahanawar/cascade_recommender/data/users.csv")

interactions = pd.read_csv(
    "/Users/prajwalmahanawar/cascade_recommender/data/interactions.csv"
)

interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])

results = []

for user_id in users["user_id"]:
    user_node = f"U{user_id}"

    recommendations = hybrid_recommend(user_node, interactions, k=10)

    for movie_id, scores in recommendations:
        results.append(
            {
                "user_id": user_node,
                "recommended_movie": movie_id,
                "hybrid_score": scores["hybrid_score"],
                "icm_score": scores["icm_score"],
                "ltm_score": scores["ltm_score"],
                "hawkes_score": scores["hawkes_score"],
            }
        )

df = pd.DataFrame(results)

print(df.head())

df.to_csv(
    "/Users/prajwalmahanawar/cascade_recommender/outputs/recommendations.csv",
    index=False,
)

print("\nRecommendations saved successfully.")
