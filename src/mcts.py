"""Determinized Information-Set MCTS over the cabt engine's search API.

This is the search half of the AlphaZero loop: the net proposes priors + a leaf value, and
this search turns them into an *improved* policy (root visit distribution) that self-play uses
as the training target. No hand-coded strategy -- only net priors, net leaf values, and the
game's own dynamics via search_begin/search_step.

Handling hidden information (honestly):
  * Every simulation resamples a plausible hidden world (`determinize.sample_determinization`)
    and starts a fresh `search_begin` rollout. Same real state -> different explored lines run
    to run; that variability is the intended hedge against uncertainty, not a bug.
  * The search only ever reads the info-hidden observation (opponent hand is None) plus the
    sampled belief. It never touches ground-truth hidden cards.

Tree children are keyed by a semantic *option signature* (kind + card/attack/target ids), not
raw option index, so "attack X at target Y" maps to the same child even when different sampled
worlds order the option list differently or make different options available (proper IS-MCTS).
Leaf value is the net's own estimate (no rollouts) -- cheap sims, more of them.
"""

from __future__ import annotations

import math

import numpy as np

from sdk_path import ensure_cg_importable

ensure_cg_importable()

from cg.api import (  # noqa: E402
    search_begin, search_step, search_release, to_observation_class,
)

from obs import parse_obs, parse_observation  # noqa: E402
from encode import encode_state, encode_option  # noqa: E402
from determinize import sample_determinization  # noqa: E402

C_PUCT = 1.5
MAX_DEPTH = 80
_DIR_EPS = 0.25
_DIR_ALPHA = 0.3


def _sig(opt) -> tuple:
    """Semantic identity of an engine Option, stable across determinizations."""
    return (
        int(opt.type), opt.attackId, opt.cardId, opt.serial, opt.area, opt.index,
        opt.inPlayArea, opt.inPlayIndex, opt.playerIndex, opt.energyIndex, opt.toolIndex,
        opt.number, opt.count, opt.specialConditionType,
    )


class Node:
    __slots__ = ("to_move", "sigs", "priors", "N", "W", "children", "expanded")

    def __init__(self):
        self.to_move = None
        self.sigs: list[tuple] = []
        self.priors = None
        self.N = None
        self.W = None
        self.children: dict[tuple, "Node"] = {}
        self.expanded = False


def _expand(node: Node, gs, sel, priors: np.ndarray):
    node.to_move = gs.your_index
    node.sigs = [_sig(lo.raw) for lo in sel.options]
    node.priors = np.asarray(priors, dtype=np.float64).copy()
    node.N = np.zeros(len(node.sigs), dtype=np.float64)
    node.W = np.zeros(len(node.sigs), dtype=np.float64)
    node.expanded = True


def _add_root_noise(node: Node):
    k = len(node.priors)
    if k == 0:
        return
    noise = np.random.dirichlet([_DIR_ALPHA] * k)
    node.priors = (1 - _DIR_EPS) * node.priors + _DIR_EPS * noise


def _build_index_list(sel, live_index: int) -> list[int]:
    """The engine wants between min_count and max_count indices. We branch on a single chosen
    option; if the select forces >=2 picks, fill the rest with other legal options."""
    if sel.min_count <= 1:
        return [live_index]
    others = [lo.index for lo in sel.options if lo.index != live_index]
    return [live_index] + others[: sel.min_count - 1]


def _select(node: Node, sel):
    """PUCT over children whose signature is available in the current (determinized) options."""
    live = {_sig(lo.raw): lo for lo in sel.options}
    avail = [i for i, s in enumerate(node.sigs) if s in live]
    if not avail:
        return None
    total = node.N.sum()
    sqrt_total = math.sqrt(total + 1.0)
    best_score, best_i = -1e30, avail[0]
    for i in avail:
        q = node.W[i] / node.N[i] if node.N[i] > 0 else 0.0
        u = C_PUCT * node.priors[i] * sqrt_total / (1.0 + node.N[i])
        score = q + u
        if score > best_score:
            best_score, best_i = score, i
    live_lo = live[node.sigs[best_i]]
    return best_i, _build_index_list(sel, live_lo.index)


def _backup(path, v_leaf: float, leaf_player: int):
    for node, i in path:
        node.N[i] += 1.0
        node.W[i] += v_leaf if node.to_move == leaf_player else -v_leaf


def _net_eval(net, gs, sel):
    sv = encode_state(gs)
    ov = np.stack([encode_option(gs, lo) for lo in sel.options])
    return net.evaluate_np(sv, ov)


def _advance_to_branch(sid, cur, depth):
    """Step through forced/empty selections (max_count==0) so the caller always sits on a real
    branching decision (or terminal). Returns (cur, gs, sel, result, depth)."""
    gs, sel = parse_observation(cur.observation)
    result = cur.observation.current.result if cur.observation.current is not None else -1
    while result == -1 and gs is not None and sel is not None and (sel.max_count == 0 or not sel.options):
        cur = search_step(sid, [])
        gs, sel = parse_observation(cur.observation)
        result = cur.observation.current.result if cur.observation.current is not None else -1
        depth += 1
        if depth > MAX_DEPTH:
            break
    return cur, gs, sel, result, depth


def _simulate(root: Node, root_observation, root_gs, deck_list, net):
    """One determinized rollout. `root` is already expanded from the real selection, so the
    tree node always corresponds to the current branching decision (empty selections are
    absorbed into the edges via _advance_to_branch)."""
    world = sample_determinization(root_gs, deck_list)
    try:
        ss = search_begin(
            root_observation, world["your_deck"], world["your_prize"],
            world["opponent_deck"], world["opponent_prize"],
            world["opponent_hand"], world["opponent_active"],
        )
    except (RuntimeError, ValueError):
        # Engine rejected this sampled world (e.g. a determinization that violates a setup
        # constraint). Skip this simulation; if every sim fails, search() falls back to net
        # priors. A rare skipped sim is acceptable and never crashes the run.
        return
    sid = ss.searchId
    path = []
    node = root
    depth = 0
    try:
        cur = ss
        while True:
            try:
                cur, gs, sel, result, depth = _advance_to_branch(sid, cur, depth)
            except (RuntimeError, ValueError):
                # Engine rejected a forced/empty-selection step in this sampled world.
                # Abandon this simulation cleanly -- no backup, so nothing is corrupted.
                return
            if result != -1 or gs is None or sel is None or depth > MAX_DEPTH:   # terminal
                if path:
                    leaf_player = path[-1][0].to_move
                    v = 0.0 if result in (2, -1) else (1.0 if result == leaf_player else -1.0)
                    _backup(path, v, leaf_player)
                return
            if not node.expanded:                              # leaf: expand + net value
                priors, value = _net_eval(net, gs, sel)
                _expand(node, gs, sel, priors)
                _backup(path, value, node.to_move)
                return
            pick = _select(node, sel)
            if pick is None:                                   # no known child available here
                _, value = _net_eval(net, gs, sel)
                _backup(path, value, node.to_move)
                return
            local_i, index_list = pick
            sig = node.sigs[local_i]
            child = node.children.get(sig)
            if child is None:
                child = Node()
                node.children[sig] = child
            try:
                cur = search_step(sid, index_list)
            except (RuntimeError, ValueError):
                # Engine rejected the step (e.g. signature collision mapped to an option that is
                # illegal in this sampled world). Abandon this simulation cleanly -- no backup,
                # so nothing is corrupted; a rare wasted sim is acceptable.
                return
            path.append((node, local_i))
            node = child
            depth += 1
    finally:
        search_release(sid)


def search(root_obs_dict: dict, net, deck_list, sims: int = 50,
           temperature: float = 1.0, add_noise: bool = False):
    """Run `sims` determinized simulations from the current real decision.

    Returns (policy, choice, index_list, selection):
      * policy      -- np.ndarray over the root options (aligned to selection.options order),
                       the improved MCTS policy (the self-play training target).
      * choice      -- chosen root option index (into selection.options).
      * index_list  -- the actual indices to pass to battle_select / search_step.
      * selection   -- the parsed root Selection (option order matches `policy`).
    Returns None during the deck-selection phase (caller returns the 60-card deck instead)."""
    root_gs, root_sel = parse_obs(root_obs_dict)
    if root_gs is None:
        return None
    root_observation = to_observation_class(root_obs_dict)

    # Pre-expand the root from the REAL selection so root.sigs / visit counts align exactly with
    # selection.options (hence with the battle_select indices the caller must play).
    root = Node()
    priors, _ = _net_eval(net, root_gs, root_sel)
    _expand(root, root_gs, root_sel, priors)
    if add_noise:
        _add_root_noise(root)

    for _ in range(sims):
        _simulate(root, root_observation, root_gs, deck_list, net)

    visits = root.N.copy()
    if visits.sum() <= 0:
        policy = root.priors / root.priors.sum()
    elif temperature <= 1e-3:
        policy = np.zeros_like(visits)
        policy[int(np.argmax(visits))] = 1.0
    else:
        heated = visits ** (1.0 / temperature)
        policy = heated / heated.sum()

    if temperature <= 1e-3:
        choice = int(np.argmax(visits if visits.sum() > 0 else root.priors))
    else:
        choice = int(np.random.choice(len(policy), p=policy))

    index_list = _build_index_list(root_sel, root_sel.options[choice].index)
    return policy, choice, index_list, root_sel
