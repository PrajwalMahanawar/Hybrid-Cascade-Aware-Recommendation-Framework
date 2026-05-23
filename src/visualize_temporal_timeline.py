import pandas as pd
import matplotlib.pyplot as plt

interactions = pd.read_csv(
    "/Users/prajwalmahanawar/cascade_recommender/data/interactions.csv"
)

interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])

# Optional: use subset if full data is too large
interactions = interactions.head(50000)

# Count interactions per day
daily_counts = interactions.set_index("timestamp").resample("D").size()

plt.figure(figsize=(12, 6))

plt.plot(daily_counts.index, daily_counts.values, marker="o")

plt.title("Temporal Interaction Timeline")
plt.xlabel("Date")
plt.ylabel("Number of Interactions")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "/Users/prajwalmahanawar/cascade_recommender/outputs/temporal_interaction_timeline.png",
    dpi=300,
)

plt.show()

print("Saved: outputs/temporal_interaction_timeline.png")
