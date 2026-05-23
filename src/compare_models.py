import pandas as pd
import networkx as nx

from icm import independent_cascade
from ltm import linear_threshold
from hawkes import temporal_intensity
from metrics import precision_at_k, recall_at_k
from normalized_hybrid import normalized_hybrid_recommend


# -----------------------------
# Load and split data
# -----------------------------

interactions = pd.read_csv(
    "/Users/prajwalmahanawar/cascade_recommender/data/interactions.csv"
)

# interactions = interactions.head(50000)

interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
interactions = interactions.sort_values("timestamp")

split_index = int(len(interactions) * 0.8)

train = interactions.iloc[:split_index].copy()
test = interactions.iloc[split_index:].copy()

print("Train Size:", len(train))
print("Test Size:", len(test))


# -----------------------------
# Build graph from TRAIN only
# -----------------------------


def build_graph(data):
    G = nx.Graph()

    for _, row in data.iterrows():
        user = f"U{row['user_id']}"
        movie = f"M{row['movie_id']}"

        G.add_edge(user, movie, weight=row["rating"])

    return G


G = build_graph(train)

# -----------------------------
# Model recommenders
# -----------------------------


def baseline_recommend(user_id, k=3):
    if user_id not in G:
        return []

    interacted = set(G.neighbors(user_id))
    scores = {}

    for node in G.nodes():
        if str(node).startswith("M") and node not in interacted:
            scores[node] = G.degree(node)

    return [
        item
        for item, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    ]


def icm_recommend(user_id, k=3):
    if user_id not in G:
        return []

    interacted = set(G.neighbors(user_id))
    scores = {}

    for node in G.nodes():
        if str(node).startswith("M") and node not in interacted:
            spread_size, _ = independent_cascade(G, node, probability=0.3)
            scores[node] = spread_size

    return [
        item
        for item, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    ]


def ltm_recommend(user_id, k=3):
    if user_id not in G:
        return []

    interacted = set(G.neighbors(user_id))
    scores = {}

    for node in G.nodes():
        if str(node).startswith("M") and node not in interacted:
            spread_size, _ = linear_threshold(G, node, threshold=0.5)
            scores[node] = spread_size

    return [
        item
        for item, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    ]


def hawkes_recommend(user_id, k=3):
    if user_id not in G:
        return []

    interacted = set(G.neighbors(user_id))
    scores = {}

    for node in G.nodes():
        if str(node).startswith("M") and node not in interacted:
            scores[node] = temporal_intensity(node, train)

    return [
        item
        for item, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    ]


def hybrid_recommend(user_id, k=3):
    if user_id not in G:
        return []

    interacted = set(G.neighbors(user_id))
    scores = {}

    for node in G.nodes():
        if str(node).startswith("M") and node not in interacted:
            icm_score, _ = independent_cascade(G, node, probability=0.3)

            ltm_score, _ = linear_threshold(G, node, threshold=0.5)

            hawkes_score = temporal_intensity(node, train)

            hybrid_score = 0.4 * icm_score + 0.2 * ltm_score + 0.4 * hawkes_score

            scores[node] = hybrid_score

    return [
        item
        for item, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    ]


# -----------------------------
# Evaluation function
# -----------------------------


def evaluate_model(model_name, recommend_function, k=3):
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

        recommended_items = recommend_function(user_node, k=k)

        if len(recommended_items) == 0:
            continue

        if any(item in actual_items for item in recommended_items):
            hits += 1

        precision = precision_at_k(recommended_items, actual_items, k=k)

        recall = recall_at_k(recommended_items, actual_items, k=k)

        precision_scores.append(precision)
        recall_scores.append(recall)

        total += 1

    hit_rate = hits / total if total > 0 else 0

    avg_precision = (
        sum(precision_scores) / len(precision_scores) if precision_scores else 0
    )

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0

    return {
        "Model": model_name,
        "Hit Rate": round(hit_rate, 4),
        f"Precision@{k}": round(avg_precision, 4),
        f"Recall@{k}": round(avg_recall, 4),
    }


# -----------------------------
# Compare all models
# -----------------------------

results = []

results.append(evaluate_model("Baseline", baseline_recommend, k=3))

results.append(evaluate_model("ICM", icm_recommend, k=3))

results.append(evaluate_model("LTM", ltm_recommend, k=3))

results.append(evaluate_model("Hawkes", hawkes_recommend, k=3))

results.append(evaluate_model("Hybrid", hybrid_recommend, k=3))

results.append(
    evaluate_model(
        "Normalized Hybrid",
        lambda user, k=3: normalized_hybrid_recommend(user, train, k),
        k=3,
    )
)

results_df = pd.DataFrame(results)

print("\nModel Comparison Results:")
print(results_df)

results_df.to_csv(
    "/Users/prajwalmahanawar/cascade_recommender/outputs/model_comparison_normalized_fullmoviedata.csv",
    index=False,
)
