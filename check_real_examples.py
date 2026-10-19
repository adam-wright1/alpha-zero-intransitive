import sys
sys.path.append('.')
import pickle
import numpy as np
from intransitive.IntransitiveGame import Intransitive
from intransitive.pytorch.NNet import NNetWrapper

game = Intransitive()
nnet = NNetWrapper(game)
nnet.load_checkpoint(folder='./temp_farmshare_v21/', filename='temp.pth.tar')

with open('temp_farmshare_v21/checkpoint_6.pth.tar.examples', 'rb') as f:
    trainExamplesHistory = pickle.load(f)

# flatten all examples from all iterations into one list
all_examples = []
for iteration_examples in trainExamplesHistory:
    all_examples.extend(iteration_examples)

print(f"Total examples available: {len(all_examples)}")

# pick a handful of random real examples and compare network output to the stored training target
np.random.seed(42)
sample_indices = np.random.choice(len(all_examples), size=5, replace=False)

for idx in sample_indices:
    board, stored_pi, stored_v = all_examples[idx]

    pred_pi, pred_v = nnet.predict(board)

    print(f"\n--- Example {idx} ---")
    print(f"Stored training value: {stored_v:.4f} | Network's predicted value: {pred_v[0]:.4f}")

    stored_max_idx = int(np.argmax(stored_pi))
    pred_max_idx = int(np.argmax(pred_pi))

    print(f"Stored pi — top action: {stored_max_idx} (prob {stored_pi[stored_max_idx]:.4f})")
    print(f"Predicted pi — top action: {pred_max_idx} (prob {pred_pi[pred_max_idx]:.4f})")
    print(f"Do top actions match? {stored_max_idx == pred_max_idx}")
    print(f"Predicted pi max probability overall: {pred_pi.max():.6f}")