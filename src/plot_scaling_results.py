import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load result files
# -----------------------------

results_10k = pd.read_csv(
    "/Users/prajwalmahanawar/cascade_recommender/outputs/model_comparison_normalized_moviedata.csv"
)

results_25k = pd.read_csv(
    "/Users/prajwalmahanawar/cascade_recommender/outputs/model_comparison_normalized_25000moviedata.csv"
)

results_50k = pd.read_csv(
    "/Users/prajwalmahanawar/cascade_recommender/outputs/model_comparison_normalized_50000moviedata.csv"
)

# -----------------------------
# Add dataset labels
# -----------------------------

results_10k["Dataset"] = "10K"
results_25k["Dataset"] = "25K"
results_50k["Dataset"] = "50K"

# -----------------------------
# Combine all
# -----------------------------

all_results = pd.concat([results_10k, results_25k, results_50k], ignore_index=True)

print(all_results)

# -----------------------------
# HIT RATE SCALING
# -----------------------------

plt.figure(figsize=(10, 6))

for model in all_results["Model"].unique():
    model_data = all_results[all_results["Model"] == model]

    plt.plot(model_data["Dataset"], model_data["Hit Rate"], marker="o", label=model)

plt.title("Hit Rate Scaling Across Dataset Sizes")
plt.xlabel("Dataset Size")
plt.ylabel("Hit Rate")

plt.legend()

plt.tight_layout()

plt.savefig("/Users/prajwalmahanawar/cascade_recommender/outputs/hit_rate_scaling.png")

plt.show()

# -----------------------------
# PRECISION SCALING
# -----------------------------

plt.figure(figsize=(10, 6))

for model in all_results["Model"].unique():
    model_data = all_results[all_results["Model"] == model]

    plt.plot(model_data["Dataset"], model_data["Precision@3"], marker="o", label=model)

plt.title("Precision@3 Scaling Across Dataset Sizes")
plt.xlabel("Dataset Size")
plt.ylabel("Precision@3")

plt.legend()

plt.tight_layout()

plt.savefig("/Users/prajwalmahanawar/cascade_recommender/outputs/precision_scaling.png")

plt.show()

# -----------------------------
# RECALL SCALING
# -----------------------------

plt.figure(figsize=(10, 6))

for model in all_results["Model"].unique():
    model_data = all_results[all_results["Model"] == model]

    plt.plot(model_data["Dataset"], model_data["Recall@3"], marker="o", label=model)

plt.title("Recall@3 Scaling Across Dataset Sizes")
plt.xlabel("Dataset Size")
plt.ylabel("Recall@3")

plt.legend()

plt.tight_layout()

plt.savefig("/Users/prajwalmahanawar/cascade_recommender/outputs/recall_scaling.png")

plt.show()

print("\nScaling plots generated successfully.")
