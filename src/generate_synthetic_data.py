import pandas as pd
import random
from datetime import datetime, timedelta

random.seed(42)

num_users = 100
num_briefs = 50
num_interactions = 5000

users = [f"U{i}" for i in range(1, num_users + 1)]
briefs = [f"B{i}" for i in range(1, num_briefs + 1)]

actions = ["view", "like", "save", "share", "submit"]

action_weights = {
    "view": 0.45,
    "like": 0.25,
    "save": 0.15,
    "share": 0.10,
    "submit": 0.05,
}

start_time = datetime(2026, 5, 1, 9, 0, 0)

rows = []

for i in range(1, num_interactions + 1):
    user = random.choice(users)
    brief = random.choice(briefs)
    action = random.choices(actions, weights=[action_weights[a] for a in actions], k=1)[
        0
    ]

    timestamp = start_time + timedelta(minutes=random.randint(0, 30000))

    rows.append([i, user, brief, action, timestamp])

interactions = pd.DataFrame(
    rows, columns=["interaction_id", "user_id", "brief_id", "action", "timestamp"]
)

users_df = pd.DataFrame(
    {
        "user_id": users,
        "user_type": [
            random.choice(["student", "developer", "designer", "researcher", "manager"])
            for _ in users
        ],
        "department": [
            random.choice(["AI", "Finance", "Healthcare", "Climate", "Marketing"])
            for _ in users
        ],
    }
)

briefs_df = pd.DataFrame(
    {
        "brief_id": briefs,
        "brand": [
            random.choice(["Nike", "Pfizer", "Stripe", "Tesla", "Google", "Meta"])
            for _ in briefs
        ],
        "name": [f"Challenge Brief {i}" for i in range(1, num_briefs + 1)],
        "start_date": ["2026-05-01"] * num_briefs,
        "end_date": ["2026-06-30"] * num_briefs,
        "created_at": ["2026-04-25 10:00:00"] * num_briefs,
        "created_by": [random.choice(users) for _ in briefs],
    }
)

users_df.to_csv(
    "/Users/prajwalmahanawar/cascade_recommender/data/users.csv", index=False
)
briefs_df.to_csv(
    "/Users/prajwalmahanawar/cascade_recommender/data/briefs.csv", index=False
)
interactions.to_csv(
    "/Users/prajwalmahanawar/cascade_recommender/data/interactions.csv", index=False
)

print("Synthetic data generated successfully")
print("Users:", len(users_df))
print("Briefs:", len(briefs_df))
print("Interactions:", len(interactions))
