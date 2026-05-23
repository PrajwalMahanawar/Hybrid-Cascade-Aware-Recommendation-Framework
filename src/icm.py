import random
import networkx as nx


def independent_cascade(graph, seed_node, probability=0.2, steps=3):
    active = {seed_node}
    newly_active = {seed_node}

    for _ in range(steps):
        next_active = set()

        for node in newly_active:
            for neighbor in graph.neighbors(node):
                if neighbor not in active:
                    if random.random() < probability:
                        next_active.add(neighbor)

        newly_active = next_active
        active.update(newly_active)

    return len(active), active
