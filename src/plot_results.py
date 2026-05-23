import pandas as pd
import matplotlib.pyplot as plt

# Load REAL experiment results
results = pd.read_csv(
    "/Users/prajwalmahanawar/cascade_recommender/outputs/model_comparison.csv"
)

print(results)

# -----------------------------
# Hit Rate Plot
# -----------------------------

plt.figure(figsize=(8, 5))

plt.bar(results["Model"], results["Hit Rate"])

plt.title("Model Comparison: Hit Rate")
plt.xlabel("Model")
plt.ylabel("Hit Rate")

plt.tight_layout()

plt.savefig(
    "/Users/prajwalmahanawar/cascade_recommender/outputs/hit_rate_comparison.png"
)

plt.show()

# -----------------------------
# Precision Plot
# -----------------------------

plt.figure(figsize=(8, 5))

plt.bar(results["Model"], results["Precision@3"])

plt.title("Model Comparison: Precision@3")
plt.xlabel("Model")
plt.ylabel("Precision@3")

plt.tight_layout()

plt.savefig(
    "/Users/prajwalmahanawar/cascade_recommender/outputs/precision_at_3_comparison.png"
)

plt.show()

# -----------------------------
# Recall Plot
# -----------------------------

plt.figure(figsize=(8, 5))

plt.bar(results["Model"], results["Recall@3"])

plt.title("Model Comparison: Recall@3")
plt.xlabel("Model")
plt.ylabel("Recall@3")

plt.tight_layout()

plt.savefig(
    "/Users/prajwalmahanawar/cascade_recommender/outputs/recall_at_3_comparison.png"
)

plt.show()

print("\nPlots generated successfully.")
