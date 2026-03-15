import numpy as np

scores = np.load("results/mirai3/scores.npy")
labels = np.load("results/mirai3/labels.npy")

print(f"Total eval rows : {len(scores)}")
print(f"Benign  (label=0): {(labels==0).sum()}")
print(f"Malicious (label=1): {(labels==1).sum()}")
print(f"\nScore stats:")
print(f"  Min   : {scores.min():.6f}")
print(f"  Max   : {scores.max():.6f}")
print(f"  Mean  : {scores.mean():.6f}")
print(f"  Median: {np.median(scores):.6f}")
print(f"\nFirst 10 scores: {scores[:10]}")
print(f"Last  10 scores: {scores[-10:]}")