"""Training step: fit the net to the MCTS visit policies + shaped returns (AlphaZero loss).

The targets come ONLY from search results (visit distributions) and game-derived shaped returns
-- never hand-tuned -- so the net is a distillation of what search discovered.
"""

from __future__ import annotations

from net import policy_value_loss


def train(net, buffer, optimizer, steps: int, batch_size: int):
    """Run `steps` minibatches from the replay buffer. Returns mean (loss, policy, value)."""
    net.train()
    tot = pol = val = 0.0
    for _ in range(steps):
        states, options, mask, target_policy, target_value = buffer.sample_batch(batch_size)
        logits, values = net.forward_batch(states, options, mask)
        loss, pl, vl = policy_value_loss(logits, values, target_policy, target_value, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot += loss.item()
        pol += pl
        val += vl
    n = max(steps, 1)
    return tot / n, pol / n, val / n
