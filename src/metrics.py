def precision_at_k(recommended_items, actual_items, k=5):

    recommended_k = recommended_items[:k]

    relevant = len(set(recommended_k).intersection(set(actual_items)))

    return relevant / k


def recall_at_k(recommended_items, actual_items, k=5):

    recommended_k = recommended_items[:k]

    relevant = len(set(recommended_k).intersection(set(actual_items)))

    if len(actual_items) == 0:
        return 0

    return relevant / len(actual_items)
