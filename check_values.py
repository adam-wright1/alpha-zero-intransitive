# this is AI slop
import pickle

with open('temp_farmshare_v21/checkpoint_6.pth.tar.examples', 'rb') as f:
    trainExamplesHistory = pickle.load(f)

# trainExamplesHistory is a list of deques, one per iteration
all_values = []
for iteration_examples in trainExamplesHistory:
    for (board, pi, v) in iteration_examples:
        all_values.append(v)

import numpy as np
all_values = np.array(all_values)
print(f"Count: {len(all_values)}")
print(f"Min: {all_values.min()}, Max: {all_values.max()}")
print(f"Mean: {all_values.mean()}, Std: {all_values.std()}")
print(f"Any NaN? {np.isnan(all_values).any()}")
print(f"Any Inf? {np.isinf(all_values).any()}")
print(f"Values > 2 (outside normal ±1-ish range): {(np.abs(all_values) > 2).sum()}")

# --- pi vector analysis ---
all_pis = []
for iteration_examples in trainExamplesHistory:
    for (board, pi, v) in iteration_examples:
        all_pis.append(pi)

all_pis = np.array(all_pis)
print(f"\n--- pi vector analysis ---")
print(f"Count: {len(all_pis)}")

max_probs = all_pis.max(axis=1)
print(f"Max probability per example — min: {max_probs.min():.6f}, max: {max_probs.max():.6f}, mean: {max_probs.mean():.6f}")

# entropy per example (higher = more uniform/less confident)
eps = 1e-12
entropies = -np.sum(all_pis * np.log(all_pis + eps), axis=1)
uniform_entropy = np.log(all_pis.shape[1])  # max possible entropy (fully uniform over 648 actions)
print(f"Entropy — min: {entropies.min():.4f}, max: {entropies.max():.4f}, mean: {entropies.mean():.4f}")
print(f"Uniform-distribution entropy (reference, higher=less confident): {uniform_entropy:.4f}")

# how many examples are close to uniform (low confidence)?
near_uniform_count = (max_probs < 0.01).sum()
print(f"Examples with max probability < 0.01 (near-uniform/low-confidence): {near_uniform_count} / {len(all_pis)} ({100*near_uniform_count/len(all_pis):.1f}%)")