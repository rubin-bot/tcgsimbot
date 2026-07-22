"""Torch-free NumPy forward pass for PVNet, used only by the packaged submission (torch is a
training-only dependency, never bundled -- see CLAUDE.md's 197.7 MiB limit).

Mirrors `net.PVNet.forward_batch`/`evaluate_np` exactly (same weights, same layer order) so
`mcts.search()` can use a `NumpyPVNet` in place of a `PVNet` with zero changes -- it only ever
calls `.evaluate_np(state_vec, option_vecs)`.
"""

from __future__ import annotations

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """x: (..., in); w: (out, in) (torch layout); b: (out,)."""
    return x @ w.T + b


class NumpyPVNet:
    def __init__(self, weights: dict, state_dim: int, option_dim: int, hidden: int):
        self.w = weights
        self.state_dim = state_dim
        self.option_dim = option_dim
        self.hidden = hidden

    @classmethod
    def load(cls, path: str) -> "NumpyPVNet":
        npz = np.load(path)
        state_dim, option_dim, hidden = (int(x) for x in npz["__config__"])
        weights = {k: npz[k] for k in npz.files if k != "__config__"}
        return cls(weights, state_dim, option_dim, hidden)

    def evaluate_np(self, state_vec: np.ndarray, option_vecs: np.ndarray):
        """Single position. state_vec (S,), option_vecs (K, O).
        Returns (priors (K,) summing to 1, value float in (-1, 1)) -- same contract as
        `net.PVNet.evaluate_np`."""
        w = self.w
        s = np.ascontiguousarray(state_vec, dtype=np.float32)
        o = np.ascontiguousarray(option_vecs, dtype=np.float32)

        embed = _relu(_linear(s, w["torso.0.weight"], w["torso.0.bias"]))
        embed = _relu(_linear(embed, w["torso.2.weight"], w["torso.2.bias"]))

        v = _relu(_linear(embed, w["value_head.0.weight"], w["value_head.0.bias"]))
        v = _linear(v, w["value_head.2.weight"], w["value_head.2.bias"])
        value = float(np.tanh(v)[0])

        k = o.shape[0]
        embed_exp = np.broadcast_to(embed, (k, self.hidden))
        pol_in = np.concatenate([embed_exp, o], axis=-1)          # (K, H+O)
        h = _relu(_linear(pol_in, w["policy_head.0.weight"], w["policy_head.0.bias"]))
        logits = _linear(h, w["policy_head.2.weight"], w["policy_head.2.bias"]).squeeze(-1)  # (K,)

        logits = logits - logits.max()
        exp = np.exp(logits)
        priors = exp / exp.sum()
        return priors, value
