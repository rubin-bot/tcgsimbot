# Ladder autopsy report

measure_state.json: checked_at=2026-07-23T08:32:57.763798+00:00, mu=568.6, rank=3664, submission_count=2
episode datasets scanned since None: [] (not yet published: [], failed: [])
episodes found per day: {'2026-07-23': 8, '2026-07-22': 67}

## Parse summary
parsed OK: 75  |  not-ours (unexpected if >0): 1  |  failed to parse: 0

## (a) Overall record
games: 75  |  wins: 34  |  losses: 41  |  draws: 0  |  errored/non-DONE episodes: 0
win rate (draws=0.5): 0.453

## (b) Loss taxonomy
  other_attrition: 34 (82.9% of losses)
  bench_wipe: 7 (17.1% of losses)

## Contributing-factor stats across losses (tools/loss_review.py, reused)
Energy-routing weight-imbalance/horizon-blind split and near-tie rate: N/A on ladder data (need evaluate() scores Kaggle never records -- see module docstring's Known Limitations). See (d)/(e) below for what IS measurable.

## (c) Top-2 failure-mode walkthroughs
(shipped weights for reference, C:\Users\RUBIN\Desktop\Projects_Cursor\Pokemon_TCG\runs\tune_run1\winner_weights.json: {'prize_diff': -0.3689932858950775, 'hp_frac_diff': 1.2161425041888685, 'active_hp_frac_diff': 1.8994005532234202, 'attacker_energy_progress': 0.39156735231784057, 'attacker_ready_and_active': 1.9580549911155, 'evolution_progress': 0.45156339567670356, 'hand_diff': 0.7732577173425174, 'cape_attached': -0.3454080506146236, 'stadium_active': 0.42480785084837314, 'ex_matchup_bonus': 1.6639670180001402, 'deck_out_risk': -0.7784991145209187, 'opp_can_ko_our_active': -0.7154635842473763, 'we_threaten_ko': 3.710155312852523, 'prize_race_delta': 0.739553800857339, 'exposed_investment': -5.61560734219277, 'best_bench_attacker_readiness': 0.19713357474989024, 'bench_attacker_advantage_bonus': 3.553697410474011, 'damage_dealt_this_turn': -0.5232132187320051})

### Bucket: other_attrition
game 87435268 vs AibePC: 12 logged decisions, terminal={'our_status': 'DONE', 'our_deck_count': 44, 'opp_deck_count': 33, 'our_prizes_remaining': 5, 'opp_prizes_remaining': 5, 'our_active_present': True, 'our_bench_count': 0}
    [game 87435268 seed None turn 4.2] mid-game decision in a other_attrition loss
      you_active: card=105 serial=65 hp=70/70 energies=[3, 3]
      opp_active: card=673 serial=3 hp=50/80 energies=[]
      - idx=0 evolve card=51 target=105(serial 65) <== CHOSEN
      - idx=1 retreat
      - idx=2 end
game 87435806 vs nimous: 5 logged decisions, terminal={'our_status': 'DONE', 'our_deck_count': 45, 'opp_deck_count': 45, 'our_prizes_remaining': 6, 'opp_prizes_remaining': 6, 'our_active_present': True, 'our_bench_count': 0}
    [game 87435806 seed None turn 2.2] mid-game decision in a other_attrition loss
      you_active: card=47 serial=72 hp=70/70 energies=[3]
      opp_active: card=344 serial=6 hp=170/170 energies=[0]
      - idx=0 play card=180
      - idx=1 attack <== CHOSEN
      - idx=2 end
game 87436356 vs naoki: 25 logged decisions, terminal={'our_status': 'DONE', 'our_deck_count': 43, 'opp_deck_count': 19, 'our_prizes_remaining': 2, 'opp_prizes_remaining': 6, 'our_active_present': True, 'our_bench_count': 0}
    [game 87436356 seed None turn 9.2] mid-game decision in a other_attrition loss
      you_active: card=180 serial=22 hp=80/110 energies=[3, 3, 3, 3, 3]
      opp_active: card=431 serial=85 hp=180/280 energies=[11, 11, 1]
      - idx=0 play card=105
      - idx=1 play card=105
      - idx=2 attack <== CHOSEN
      - idx=3 end

### Bucket: bench_wipe
game 87439590 vs naoteru nakamura: 10 logged decisions, terminal={'our_status': 'DONE', 'our_deck_count': 44, 'opp_deck_count': 25, 'our_prizes_remaining': 5, 'opp_prizes_remaining': 6, 'our_active_present': False, 'our_bench_count': 0}
    [game 87439590 seed None turn 4.2] mid-game decision in a bench_wipe loss
      you_active: card=105 serial=64 hp=40/70 energies=[3, 3]
      opp_active: card=673 serial=16 hp=50/80 energies=[6, 6]
      - idx=0 evolve card=51 target=105(serial 64) <== CHOSEN
      - idx=1 evolve card=51 target=105(serial 64)
      - idx=2 end
game 87442303 vs Vishwas Mishra: 12 logged decisions, terminal={'our_status': 'DONE', 'our_deck_count': 43, 'opp_deck_count': 19, 'our_prizes_remaining': 4, 'opp_prizes_remaining': 5, 'our_active_present': False, 'our_bench_count': 0}
    [game 87442303 seed None turn 3.3] mid-game decision in a bench_wipe loss
      you_active: card=51 serial=9 hp=150/150 energies=[3, 3]
      opp_active: card=673 serial=63 hp=80/80 energies=[]
      - idx=0 attack <== CHOSEN
      - idx=1 end
game 87482630 vs Zeina Shaltout: 25 logged decisions, terminal={'our_status': 'DONE', 'our_deck_count': 39, 'opp_deck_count': 11, 'our_prizes_remaining': 2, 'opp_prizes_remaining': 3, 'our_active_present': False, 'our_bench_count': 0}
    [game 87482630 seed None turn 8.5] mid-game decision in a bench_wipe loss
      you_active: none
      opp_active: card=743 serial=87 hp=110/140 energies=[5]
      - idx=0 card <== CHOSEN

## (d) Attacker-starvation target check (Dwebble pipeline hypothesis)
starvation events in ladder losses: 20  |  targeted Dwebble (pre-evolution, 344): 3  |  other target: 17
Quantitative tied-vs-weight-imbalance split: NOT recomputable from ladder data (no evaluate() scores -- see Known Limitations). Qualitative read: the shipped weights (above) put a large negative weight on `exposed_investment` and a large positive weight on `we_threaten_ko`/`bench_attacker_advantage_bonus` -- starvation decisions where those two point in opposite directions (our Crustle is exposed AND we could threaten a KO some other way) are the board states most likely to have been genuine near-ties, by the same reasoning the local dry-run's tied_and_lost examples showed.

## (e) Secondary metrics
attack-decline rate (ladder losses): 93/253 (36.8%)
evolve-decline rate (ladder losses): 25/58 (43.1%)
near-tie rate (ladder losses): N/A -- no scores in ladder data

## (f) Opponent picture
our current leaderboard score: 568.6
  vs stronger opponents: 24 games -- {'opponent_win': 18, 'candidate_win': 6}
  vs similar opponents: 21 games -- {'candidate_win': 12, 'opponent_win': 9}
  vs weaker opponents: 30 games -- {'opponent_win': 14, 'candidate_win': 16}

## Comparison vs. local dry-run (2026-07-22)
local losses analyzed: 42  |  ladder losses analyzed: 41
local attack-decline rate: 14/147 (9.5%)
local evolve-decline rate: 12/47 (25.5%)
ladder attack-decline rate: 93/253 (36.8%)
ladder evolve-decline rate: 25/58 (43.1%)

## Ranked fix candidates (evidence-based, NOT implemented this cycle)
See report body above for the evidence counts behind each of these; each would be verified via scripts/build_kernel_bakeoff.py's 400+-game kernel gate before shipping.
