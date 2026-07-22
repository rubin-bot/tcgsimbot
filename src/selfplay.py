"""Self-play game generation.

Plays full games with the current-best net driving BOTH sides via determinized MCTS, and
records, per decision, the training tuple:
    (encoded state, option features, MCTS visit policy, player-to-move).
At game end each decision gets a value target = the SHAPED RETURN from that player's
perspective:  z + alpha * prize_differential   (clipped to [-1, 1]).

z is the pure win/loss/draw outcome (the philosophy's honest target); the small alpha*prize
term is the deliberate, tunable shaping the user chose for a denser CPU-friendly signal. Set
alpha=0 to recover pure-outcome training.

Games are stored as compact ragged npz files (states + CSR-style option/policy arrays) that
replay.py samples for training. `generate()` parallelizes across processes for the 16 cores.
"""

from __future__ import annotations

import os
import random

import numpy as np

from sdk_path import ensure_cg_importable

ensure_cg_importable()

from cg.game import battle_start, battle_select, battle_finish  # noqa: E402

from obs import parse_obs  # noqa: E402
from encode import encode_state, encode_option  # noqa: E402
from mcts import search  # noqa: E402
from net import PVNet  # noqa: E402

START_PRIZES = 6


def _temperature(ply: int, temp_moves: int, hi: float, lo: float) -> float:
    return hi if ply < temp_moves else lo


def play_game(net, deck_list, sims=50, temp_moves=10, temp_hi=1.0, temp_lo=0.25, seed=None):
    """Play one self-play game. Returns (records, result, final_remaining_prizes).
    Each record = (state_vec, option_matrix, policy, player_index)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    obs, sd = battle_start(deck_list, deck_list)
    if obs is None:
        return [], -1, [START_PRIZES, START_PRIZES]

    records = []
    ply = 0
    result = -1
    final_remaining = [START_PRIZES, START_PRIZES]
    for _ in range(6000):
        st = obs.get("current")
        if st and st.get("result", -1) != -1:
            result = st["result"]
            final_remaining = [len(st["players"][0]["prize"]), len(st["players"][1]["prize"])]
            break
        gs, sel = parse_obs(obs)
        if gs is None:                       # deck-selection phase (shouldn't occur post-start)
            obs = battle_select(deck_list)
            continue
        temp = _temperature(ply, temp_moves, temp_hi, temp_lo)
        policy, choice, index_list, root_sel = search(
            obs, net, deck_list, sims=sims, temperature=temp, add_noise=True)
        state_vec = encode_state(gs)
        option_mat = np.stack([encode_option(gs, lo) for lo in sel.options]).astype(np.float32)
        records.append((state_vec, option_mat, policy.astype(np.float32), gs.your_index))
        obs = battle_select(index_list)
        ply += 1
    battle_finish()
    return records, result, final_remaining


def shaped_value(player: int, result: int, final_remaining, alpha: float) -> float:
    """z (win/loss/draw for `player`) + alpha * prize differential, clipped to [-1, 1]."""
    if result in (2, -1):
        z = 0.0
    else:
        z = 1.0 if result == player else -1.0
    taken = [START_PRIZES - final_remaining[0], START_PRIZES - final_remaining[1]]
    prize_diff = (taken[player] - taken[1 - player]) / START_PRIZES
    return float(np.clip(z + alpha * prize_diff, -1.0, 1.0))


def build_arrays(records, result, final_remaining, alpha):
    """Pack records + value targets into compact ragged arrays (CSR-style options/policies)."""
    states = np.stack([r[0] for r in records]).astype(np.float32)          # (M, S)
    counts = np.asarray([r[1].shape[0] for r in records], dtype=np.int32)  # (M,)
    options = np.concatenate([r[1] for r in records], axis=0).astype(np.float32)  # (T, O)
    policies = np.concatenate([r[2] for r in records], axis=0).astype(np.float32)  # (T,)
    values = np.asarray([shaped_value(r[3], result, final_remaining, alpha)
                         for r in records], dtype=np.float32)              # (M,)
    return {"states": states, "counts": counts, "options": options,
            "policies": policies, "values": values}


def save_game(path: str, arrays: dict):
    np.savez_compressed(path, **arrays)


def _generate_worker(net_path, deck_list, n_games, out_dir, sims, alpha, temp_moves, base_seed, wid):
    # Each worker is its own process; without this, every worker's torch would default to a
    # multi-threaded BLAS/OpenMP pool sized to ALL cores, so N workers means N x oversubscription
    # of the machine (the cause of a full CPU/RAM-exhaustion crash observed during testing).
    import torch
    torch.set_num_threads(1)
    net = PVNet.load(net_path)
    written = 0
    for g in range(n_games):
        seed = base_seed + wid * 1_000_003 + g
        try:
            records, result, final_remaining = play_game(
                net, deck_list, sims=sims, temp_moves=temp_moves, seed=seed)
        except (OSError, RuntimeError) as e:
            # A rare engine-side fault (e.g. a cg.dll access violation surfacing as an
            # OSError) must not take down the whole worker pool and lose the rest of the
            # iteration's self-play -- skip just this one game. Deliberately no
            # battle_finish() here: battle_ptr may now point at corrupted native state.
            print(f"[selfplay] worker {wid} game {g} (seed={seed}) failed: {e!r}", flush=True)
            continue
        if result == -1 or not records:
            continue
        arrays = build_arrays(records, result, final_remaining, alpha)
        save_game(os.path.join(out_dir, f"game_w{wid}_g{g}_{seed}.npz"), arrays)
        written += 1
        print(f"[selfplay] worker {wid}: game {g + 1}/{n_games} done ({written} written)",
              flush=True)
    return written


def generate(net_path, deck_list, n_games, out_dir, workers=1, sims=50, alpha=0.3,
             temp_moves=10, base_seed=0, timeout=1800):
    """Generate `n_games` self-play games (parallel across `workers` processes).

    `timeout` bounds how long we wait on a hung/deadlocked worker pool before terminating it
    and moving on -- games already written to `out_dir` (each worker saves per-game, before
    returning) are never lost even if the pool is force-killed.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Set for the parent too (covers workers<=1, which runs in-process) and so children that
    # import torch during unpickling (before _generate_worker's own set_num_threads runs)
    # inherit a capped thread pool from process creation, not just from our explicit call.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    if workers <= 1:
        return _generate_worker(net_path, deck_list, n_games, out_dir, sims, alpha,
                                temp_moves, base_seed, 0)
    from multiprocessing import Pool, TimeoutError as MPTimeoutError
    per = [n_games // workers + (1 if i < n_games % workers else 0) for i in range(workers)]
    args = [(net_path, deck_list, per[i], out_dir, sims, alpha, temp_moves, base_seed, i)
            for i in range(workers) if per[i] > 0]
    with Pool(len(args)) as pool:
        async_res = pool.starmap_async(_generate_worker, args)
        try:
            return sum(async_res.get(timeout=timeout))
        except MPTimeoutError:
            print(f"[selfplay] self-play timed out after {timeout}s ({len(args)} workers) "
                  "-- terminating hung pool, continuing with games already written to disk",
                  flush=True)
            pool.terminate()
            return 0
