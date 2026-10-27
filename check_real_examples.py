import sys
sys.path.append('.')
import pickle
import numpy as np
import torch
from intransitive.IntransitiveGame import Intransitive
from intransitive.pytorch.NNet import NNetWrapper

game = Intransitive()
nnet = NNetWrapper(game)
nnet.load_checkpoint(folder='./temp_farmshare_vXX/', filename='temp.pth.tar')  # update to new run's folder

with open('temp_farmshare_vXX/checkpoint_N.pth.tar.examples', 'rb') as f:  # update path
    trainExamplesHistory = pickle.load(f)

all_examples = []
for iteration_examples in trainExamplesHistory:
    all_examples.extend(iteration_examples)

print(f"Total examples available: {len(all_examples)}")

np.random.seed(42)
sample_indices = np.random.choice(len(all_examples), size=5, replace=False)

for idx in sample_indices:
    board, stored_pi, stored_v, stored_capture, stored_vuln = all_examples[idx]

    board_tensor = torch.FloatTensor(nnet.board_to_planes(board).astype(np.float64)).view(1, 7, 9, 9)
    nnet.nnet.eval()
    with torch.no_grad():
        log_pi, v, capture_pred, vuln_pred = nnet.nnet(board_tensor)
    pred_pi = torch.exp(log_pi).cpu().numpy()[0]
    pred_v = v.cpu().numpy()[0]
    pred_capture = capture_pred.cpu().numpy()[0]
    pred_vuln = vuln_pred.cpu().numpy()[0]

    print(f"\n--- Example {idx} ---")
    print(f"Stored training value: {stored_v:.4f} | Network's predicted value: {pred_v[0]:.4f}")

    stored_max_idx = int(np.argmax(stored_pi))
    pred_max_idx = int(np.argmax(pred_pi))
    print(f"Stored pi — top action: {stored_max_idx} (prob {stored_pi[stored_max_idx]:.4f})")
    print(f"Predicted pi — top action: {pred_max_idx} (prob {pred_pi[pred_max_idx]:.4f})")
    print(f"Do top actions match? {stored_max_idx == pred_max_idx}")

    capture_correct = np.array(stored_capture)
    capture_pred_binary = (pred_capture > 0.5).astype(int)
    capture_accuracy = (capture_correct == capture_pred_binary).mean()
    print(f"Capture-prediction accuracy on this example: {capture_accuracy:.2%}")