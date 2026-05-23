import pandas as pd

from hybrid_recommender import hybrid_recommend
from metrics import precision_at_k, recall_at_k

interactions = pd.read_csv(
    "/Users/prajwalmahanawar/cascade_recommender/data/interactions.csv"
)

interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
interactions = interactions.sort_values("timestamp")

split_index = int(len(interactions) * 0.8)

train = interactions.iloc[:split_index].copy()
test = interactions.iloc[split_index:].copy()

print("Train Size:", len(train))
print("Test Size:", len(test))

hits = 0
total = 0

precision_scores = []
recall_scores = []

test_users = test["user_id"].unique()

for user in test_users:
    user_node = f"U{user}"

    actual_items = set(
        f"M{movie_id}" for movie_id in test[test["user_id"] == user]["movie_id"]
    )

    recommendations = hybrid_recommend(user_node, train, k=3)

    recommended_items = [item for item, _ in recommendations[:3]]

    print(f"\nUser: {user_node}")
    print("Actual Future Items:", actual_items)
    print("Recommended Items:", recommended_items)

    if any(item in actual_items for item in recommended_items):
        hits += 1

    precision = precision_at_k(recommended_items, actual_items, k=3)

    recall = recall_at_k(recommended_items, actual_items, k=3)

    precision_scores.append(precision)
    recall_scores.append(recall)

    total += 1

hit_rate = hits / total if total > 0 else 0
avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0

print("\nHit Rate:", hit_rate)
print("Average Precision@3:", avg_precision)
print("Average Recall@3:", avg_recall)
