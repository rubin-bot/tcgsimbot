"""Policy/value network (training-time PyTorch).

Architecture (tiny, CPU-friendly -- see CLAUDE.md inference budget):
  torso:  state[STATE_DIM] -> MLP -> embed[H]
  value:  embed -> MLP -> scalar in (-1, 1)   (predicts the shaped return)
  policy: for each legal option, logit = MLP([embed || option_feats]); softmax over the
          option set = the move-preference distribution. Scoring options individually is how
          we handle the engine's variable, context-dependent legal-option lists.

Two entry points:
  * `evaluate_np(state_vec, option_vecs) -> (priors, value)`  -- inference/MCTS, no grad, numpy.
  * `forward_batch(states, options, mask) -> (logits, values)` -- training, padded + masked.

`export_numpy(path)` dumps the weights so the *submission* can run a torch-free NumPy forward
(torch is never bundled -- 197.7 MiB limit). torch is a training-only dependency.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from encode import STATE_DIM, OPTION_DIM

_NEG_INF = -1e9


class PVNet(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, option_dim: int = OPTION_DIM, hidden: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.option_dim = option_dim
        self.hidden = hidden
        self.torso = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1), nn.Tanh(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden + option_dim, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward_batch(self, states: torch.Tensor, options: torch.Tensor, mask: torch.Tensor):
        """states (B, S); options (B, Kmax, O); mask (B, Kmax) bool.
        Returns logits (B, Kmax) (masked to -inf where invalid) and values (B,)."""
        embed = self.torso(states)                              # (B, H)
        values = self.value_head(embed).squeeze(-1)             # (B,)
        b, kmax, _ = options.shape
        embed_exp = embed.unsqueeze(1).expand(b, kmax, self.hidden)
        pol_in = torch.cat([embed_exp, options], dim=-1)        # (B, Kmax, H+O)
        logits = self.policy_head(pol_in).squeeze(-1)           # (B, Kmax)
        logits = logits.masked_fill(~mask, _NEG_INF)
        return logits, values

    @torch.no_grad()
    def evaluate_np(self, state_vec: np.ndarray, option_vecs: np.ndarray):
        """Single position. state_vec (S,), option_vecs (K, O).
        Returns (priors (K,) summing to 1, value float in (-1, 1))."""
        self.eval()
        s = torch.from_numpy(np.ascontiguousarray(state_vec, dtype=np.float32)).unsqueeze(0)
        k = option_vecs.shape[0]
        o = torch.from_numpy(np.ascontiguousarray(option_vecs, dtype=np.float32)).unsqueeze(0)
        mask = torch.ones(1, k, dtype=torch.bool)
        logits, value = self.forward_batch(s, o, mask)
        priors = F.softmax(logits, dim=1).squeeze(0).numpy()
        return priors, float(value.item())

    def save(self, path: str):
        torch.save({"state_dict": self.state_dict(),
                    "config": {"state_dim": self.state_dim, "option_dim": self.option_dim,
                               "hidden": self.hidden}}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "PVNet":
        ckpt = torch.load(path, map_location=map_location)
        net = cls(**ckpt["config"])
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        return net

    def export_numpy(self, path: str):
        """Dump weights to .npz for a torch-free inference forward in the submission."""
        arrays = {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}
        arrays["__config__"] = np.asarray([self.state_dim, self.option_dim, self.hidden], dtype=np.int64)
        np.savez(path, **arrays)


def policy_value_loss(logits, values, target_policy, target_value, mask):
    """AlphaZero loss: policy cross-entropy (vs MCTS visit distribution) + value MSE (vs the
    shaped return). target_policy (B, Kmax) sums to 1 over valid entries; masked entries are 0."""
    log_probs = F.log_softmax(logits, dim=1)
    log_probs = log_probs.masked_fill(~mask, 0.0)               # kill -inf*0 -> nan
    policy_loss = -(target_policy * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(values, target_value)
    return policy_loss + value_loss, policy_loss.item(), value_loss.item()
