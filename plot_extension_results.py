import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("extension_results.csv")

# Accuracy vs tau for each encoder (SST only)
sst = df[df["dataset"] == "sst"]

for enc in sorted(sst["encoder"].unique()):
    sub = sst[sst["encoder"] == enc]
    plt.plot(sub["scaling_tau"], sub["test_accuracy"], marker="o", label=enc)

plt.xlabel("τ (scaling)")
plt.ylabel("Test accuracy")
plt.title("SST: Accuracy vs τ by encoder")
plt.grid(True)
plt.legend()
plt.savefig("extension_accuracy_vs_tau.png", bbox_inches="tight", dpi=300)
plt.close()

# Correlation vs tau (SST)
for enc in sorted(sst["encoder"].unique()):
    sub = sst[sst["encoder"] == enc]
    plt.plot(sub["scaling_tau"], sub["correlation_r"], marker="o", label=enc)

plt.xlabel("τ (scaling)")
plt.ylabel("Polarity–attention correlation r")
plt.title("SST: Correlation vs τ by encoder")
plt.grid(True)
plt.legend()
plt.savefig("extension_correlation_vs_tau.png", bbox_inches="tight", dpi=300)
plt.close()

# Training time by encoder (average over all configs that ran)
group = df.groupby("encoder")["training_time_sec"].mean().reset_index()

plt.bar(group["encoder"], group["training_time_sec"])
plt.xlabel("Encoder")
plt.ylabel("Avg training time (sec)")
plt.title("Average training time by encoder")
plt.grid(axis="y")
plt.savefig("extension_training_time.png", bbox_inches="tight", dpi=300)
plt.close()

print("Saved plots: extension_accuracy_vs_tau.png, extension_correlation_vs_tau.png, extension_training_time.png")
