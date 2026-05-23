import pandas as pd
import networkx as nx

from icm import independent_cascade
from ltm import linear_threshold
from hawkes import temporal_intensity
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


def build_graph(data):
    G = nx.Graph()

    for _, row in data.iterrows():
        user = f"U{row['user_id']}"
        movie = f"M{row['movie_id']}"

        G.add_edge(user, movie, weight=row["rating"])

    return G


G = build_graph(train)


def hybrid_recommend_weighted(user_id, w_icm, w_ltm, w_hawkes, k=3):
    if user_id not in G:
        return []

    interacted = set(G.neighbors(user_id))
    scores = {}

    for node in G.nodes():
        if str(node).startswith("M") and node not in interacted:
            icm_score, _ = independent_cascade(G, node, probability=0.3)

            ltm_score, _ = linear_threshold(G, node, threshold=0.5)

            hawkes_score = temporal_intensity(node, train)

            hybrid_score = (
                w_icm * icm_score + w_ltm * ltm_score + w_hawkes * hawkes_score
            )

            scores[node] = hybrid_score

    return [
        item
        for item, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    ]


def evaluate_weights(w_icm, w_ltm, w_hawkes, k=3):
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

        recommended_items = hybrid_recommend_weighted(
            user_node, w_icm, w_ltm, w_hawkes, k=k
        )

        if len(recommended_items) == 0:
            continue

        if any(item in actual_items for item in recommended_items):
            hits += 1

        precision_scores.append(precision_at_k(recommended_items, actual_items, k=k))

        recall_scores.append(recall_at_k(recommended_items, actual_items, k=k))

        total += 1

    hit_rate = hits / total if total > 0 else 0

    avg_precision = (
        sum(precision_scores) / len(precision_scores) if precision_scores else 0
    )

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0

    return hit_rate, avg_precision, avg_recall


results = []

weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for w_icm in weights:
    for w_ltm in weights:
        for w_hawkes in weights:
            if round(w_icm + w_ltm + w_hawkes, 1) == 1.0:
                hit_rate, precision, recall = evaluate_weights(
                    w_icm, w_ltm, w_hawkes, k=3
                )

                results.append(
                    {
                        "w_icm": w_icm,
                        "w_ltm": w_ltm,
                        "w_hawkes": w_hawkes,
                        "hit_rate": round(hit_rate, 4),
                        "precision_at_3": round(precision, 4),
                        "recall_at_3": round(recall, 4),
                    }
                )


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["recall_at_3", "precision_at_3", "hit_rate"], ascending=False
)

print("\nTop Hybrid Weight Results:")
print(results_df.head(10))

results_df.to_csv(
    "/Users/prajwalmahanawar/cascade_recommender/outputs/hybrid_weight_tuning_results.csv",
    index=False,
)
