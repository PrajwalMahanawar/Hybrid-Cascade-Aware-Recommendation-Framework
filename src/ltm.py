def linear_threshold(graph, seed_node, threshold=0.5, steps=3):
    active = {seed_node}

    for _ in range(steps):
        new_active = set(active)

        for node in graph.nodes():
            if node not in active:
                neighbors = list(graph.neighbors(node))

                if not neighbors:
                    continue

                active_neighbors = [n for n in neighbors if n in active]
                influence_ratio = len(active_neighbors) / len(neighbors)

                if influence_ratio >= threshold:
                    new_active.add(node)

        if new_active == active:
            break

        active = new_active

    return len(active), active
