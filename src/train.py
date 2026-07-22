"""Self-play training orchestrator -- one resumable command.

Each iteration:
  1. self-play: generate games with the current GENERATOR net (best-so-far), parallel workers.
  2. train:     update the LEARNER net on the recent replay buffer (MCTS policy + shaped return).
  3. eval:      learner vs a FIXED reference set (rule-based baseline + frozen snapshots) -> the
                trustworthy progress curve (win% must rise here, not vs the moving opponent).
  4. gate:      if the learner beats the generator head-to-head by a margin, promote it to the
                new generator and snapshot it into the frozen pool (philosophy: only promote on
                a clear margin, so self-play doesn't regress into an exploitable cycle).

Artifacts (all gitignored) live under runs/<name>/: best.pt (generator), learner.pt (+opt),
pool/ckpt_*.pt (frozen opponents), metrics.csv, state.json. Resume with the same --name; the
loop is safe to Ctrl-C between iterations.

Alpha (reward shaping) anneals toward 0 across iterations -- dense early signal, honest late.

Usage (fast-feedback defaults ~minutes/iter on CPU):
  .venv/Scripts/python src/train.py --name run1 --iters 20
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time

import torch

from sdk_path import ensure_cg_importable

ensure_cg_importable()

from net import PVNet  # noqa: E402
from selfplay import generate  # noqa: E402
from replay import ReplayBuffer  # noqa: E402
from train_step import train  # noqa: E402
from evaluate import evaluate, win_rate, make_net_agent  # noqa: E402
from baseline import agent as baseline_agent  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_deck(path):
    return [int(x) for x in open(path).read().split("\n")[:60]]


def alpha_at(it, alpha0, anneal_iters):
    if anneal_iters <= 0:
        return alpha0
    return float(alpha0 * max(0.0, 1.0 - it / anneal_iters))


def clone(net):
    return copy.deepcopy(net).eval()


def _next_ckpt_path(pool_dir):
    """Next unique ckpt_N.pt path, scanning existing files so the id keeps climbing even
    after pool_files (the in-memory list) gets truncated to --max-pool entries -- using
    len(pool_files) for the id would collide with an already-truncated list and silently
    overwrite an old snapshot."""
    ids = []
    for name in os.listdir(pool_dir):
        if name.startswith("ckpt_") and name.endswith(".pt"):
            try:
                ids.append(int(name[len("ckpt_"):-len(".pt")]))
            except ValueError:
                pass
    next_id = (max(ids) + 1) if ids else 0
    return os.path.join(pool_dir, f"ckpt_{next_id}.pt")


class Run:
    def __init__(self, root, hidden):
        self.root = root
        self.data_dir = os.path.join(root, "selfplay_data")
        self.pool_dir = os.path.join(root, "pool")
        self.best_path = os.path.join(root, "best.pt")
        self.learner_path = os.path.join(root, "learner.pt")
        self.opt_path = os.path.join(root, "opt.pt")
        self.state_path = os.path.join(root, "state.json")
        self.metrics_path = os.path.join(root, "metrics.csv")
        for d in (root, self.data_dir, self.pool_dir):
            os.makedirs(d, exist_ok=True)
        self.hidden = hidden

    def exists(self):
        return os.path.exists(self.state_path)

    def load_state(self):
        with open(self.state_path) as f:
            return json.load(f)

    def save_state(self, state):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, self.state_path)

    def log_metrics(self, row, header):
        new = not os.path.exists(self.metrics_path)
        with open(self.metrics_path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(header)
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--games-per-iter", type=int, default=24)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--eval-sims", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train-steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eval-games", type=int, default=16)
    ap.add_argument("--gate-games", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=5,
                    help="run the (expensive net-vs-net) eval + gate only every N iters")
    ap.add_argument("--gate-thresh", type=float, default=0.55)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha0", type=float, default=0.3)
    ap.add_argument("--alpha-anneal-iters", type=int, default=15)
    ap.add_argument("--buffer-games", type=int, default=200)
    ap.add_argument("--snapshot-every", type=int, default=3)
    ap.add_argument("--max-pool", type=int, default=4)
    ap.add_argument("--deck", default=os.path.join(REPO, "decks", "baseline_deck.csv"))
    ap.add_argument("--sanity-deck", default=None,
                    help="optional second deck (different archetype) for a non-mirror eval "
                         "opponent, so the win-rate signal can't be fooled by a policy that "
                         "only works against a mirror of its own deck")
    args = ap.parse_args()

    run = Run(os.path.join(REPO, "runs", args.name), args.hidden)
    deck = load_deck(args.deck)
    sanity_deck = load_deck(args.sanity_deck) if args.sanity_deck else None

    learner = PVNet(hidden=args.hidden)
    opt = torch.optim.Adam(learner.parameters(), lr=args.lr)

    if run.exists():
        state = run.load_state()
        learner = PVNet.load(run.learner_path)
        opt = torch.optim.Adam(learner.parameters(), lr=args.lr)
        if os.path.exists(run.opt_path):
            opt.load_state_dict(torch.load(run.opt_path))
        pool_files = state["pool"]
        start_iter = state["iteration"]
        print(f"[resume] {args.name} @ iter {start_iter}, pool={len(pool_files)}")
    else:
        learner.save(run.learner_path)
        learner.save(run.best_path)                      # generator := random init
        snap0 = os.path.join(run.pool_dir, "ckpt_0.pt")
        learner.save(snap0)                              # frozen reference: iter-0 net
        pool_files = [snap0]
        start_iter = 0
        run.save_state({"iteration": 0, "pool": pool_files})
        print(f"[new run] {args.name} at {run.root}")

    for it in range(start_iter, args.iters):
        t0 = time.time()
        alpha = alpha_at(it, args.alpha0, args.alpha_anneal_iters)

        # 1) self-play with the generator (best.pt)
        n = generate(run.best_path, deck, args.games_per_iter, run.data_dir,
                     workers=args.workers, sims=args.sims, alpha=alpha, base_seed=it * 100_003)
        _prune_data(run.data_dir, args.buffer_games)

        # 2) train the learner on recent games
        buf = ReplayBuffer(run.data_dir, max_games=args.buffer_games)
        loss, pl, vl = train(learner, buf, opt, args.train_steps, args.batch)

        # 3+4) eval vs fixed reference + gated promotion -- PERIODIC (net-vs-net games are the
        # main cost and are pure measurement, so most iters skip them and just keep learning).
        do_eval = (it % args.eval_every == 0) or (it == args.iters - 1)
        wr_b_s = wr_p_s = gate_s = wr_s_s = ""
        promoted = False
        if do_eval:
            opponents = {"baseline": baseline_agent}
            opponent_decks = {}
            for i, pf in enumerate(pool_files):
                opponents[f"ckpt{i}"] = make_net_agent(PVNet.load(pf), deck, sims=args.eval_sims)
            if sanity_deck is not None:
                opponents["sanity"] = baseline_agent
                opponent_decks["sanity"] = sanity_deck
            ev = evaluate(learner, opponents, deck, n_games=args.eval_games, sims=args.eval_sims,
                          base_seed=it * 7 + 1, opponent_decks=opponent_decks)
            wr_baseline = ev["baseline"]
            pool_wrs = [ev[k] for k in ev if k.startswith("ckpt")]
            wr_pool = sum(pool_wrs) / len(pool_wrs)
            wr_sanity = ev.get("sanity")
            gen = PVNet.load(run.best_path)
            gate = win_rate(make_net_agent(learner, deck, sims=args.eval_sims),
                            make_net_agent(gen, deck, sims=args.eval_sims),
                            deck, deck, args.gate_games, base_seed=it * 13 + 5)
            promoted = gate >= args.gate_thresh
            if promoted:                                  # promote learner -> generator + snapshot
                learner.save(run.best_path)
                snap = _next_ckpt_path(run.pool_dir)
                learner.save(snap)
                pool_files.append(snap)
                if len(pool_files) > args.max_pool:
                    pool_files = [pool_files[0]] + pool_files[-(args.max_pool - 1):]
            wr_b_s, wr_p_s, gate_s = f"{wr_baseline:.3f}", f"{wr_pool:.3f}", f"{gate:.3f}"
            wr_s_s = f"{wr_sanity:.3f}" if wr_sanity is not None else ""

        # persist (resumable, Ctrl-C safe between iters)
        learner.save(run.learner_path)
        torch.save(opt.state_dict(), run.opt_path)
        run.save_state({"iteration": it + 1, "pool": pool_files})

        dt = time.time() - t0
        run.log_metrics(
            [it, f"{alpha:.3f}", n, f"{loss:.3f}", f"{pl:.3f}", f"{vl:.3f}",
             wr_b_s, wr_p_s, wr_s_s, gate_s, int(promoted), f"{dt:.1f}"],
            ["iter", "alpha", "games", "loss", "policy", "value",
             "wr_baseline", "wr_pool", "wr_sanity", "gate", "promoted", "sec"])
        evtxt = (f"wr_base={wr_b_s} wr_pool={wr_p_s} wr_sanity={wr_s_s} gate={gate_s} "
                 f"{'PROMOTED' if promoted else ''}") if do_eval else "(train-only)"
        print(f"iter {it:3d} | a={alpha:.2f} games={n} loss={loss:.3f} {evtxt} | {dt:.0f}s")

    print(f"done. metrics -> {run.metrics_path}")


def _prune_data(data_dir, keep):
    """Keep only the most recent `keep` game files (sliding replay window)."""
    from replay import list_games
    files = list_games(data_dir)
    for f in files[:-keep] if keep and len(files) > keep else []:
        try:
            os.remove(f)
        except OSError:
            pass


if __name__ == "__main__":
    main()
