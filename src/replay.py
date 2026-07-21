"""Replay buffer over self-play games.

Loads the most recent npz games written by selfplay.py, flattens them to per-decision samples,
and collates random batches into padded + masked tensors for training (variable option counts
padded to Kmax per batch).
"""

from __future__ import annotations

import glob
import os
import random

import numpy as np
import torch


def list_games(data_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(data_dir, "*.npz")))


def collate(batch):
    """batch of (state (S,), options (K,O), policy (K,), value) -> padded tensors."""
    b = len(batch)
    s_dim = batch[0][0].shape[0]
    o_dim = batch[0][1].shape[1]
    kmax = max(x[1].shape[0] for x in batch)
    states = np.zeros((b, s_dim), np.float32)
    options = np.zeros((b, kmax, o_dim), np.float32)
    mask = np.zeros((b, kmax), dtype=bool)
    target_policy = np.zeros((b, kmax), np.float32)
    target_value = np.zeros((b,), np.float32)
    for i, (s, o, p, v) in enumerate(batch):
        k = o.shape[0]
        states[i] = s
        options[i, :k] = o
        mask[i, :k] = True
        target_policy[i, :k] = p
        target_value[i] = v
    return (torch.from_numpy(states), torch.from_numpy(options), torch.from_numpy(mask),
            torch.from_numpy(target_policy), torch.from_numpy(target_value))


class ReplayBuffer:
    def __init__(self, data_dir: str, max_games: int | None = None):
        self.data_dir = data_dir
        self.max_games = max_games
        self.samples: list[tuple] = []
        self.reload()

    def reload(self):
        files = list_games(self.data_dir)
        if self.max_games:
            files = files[-self.max_games:]
        samples = []
        for f in files:
            try:
                d = np.load(f)
            except (OSError, ValueError):
                continue                                  # a game still being written; skip
            states, counts = d["states"], d["counts"]
            options, policies, values = d["options"], d["policies"], d["values"]
            off = 0
            for i in range(len(states)):
                k = int(counts[i])
                samples.append((states[i], options[off:off + k], policies[off:off + k],
                                float(values[i])))
                off += k
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def sample_batch(self, batch_size: int):
        batch = random.sample(self.samples, min(batch_size, len(self.samples)))
        return collate(batch)
