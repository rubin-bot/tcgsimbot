# cg SDK — verified ground truth (captured on Linux x86_64, this session, 2026-07-21)

Source: Kaggle competition `pokemon-tcg-ai-battle`, path
`sample_submission/sample_submission/`. Downloaded selectively (PDF card-image scans and
JP CSV skipped; ~1.7 MB total vs. the ~660 MB of the full bundle). Card text from
`pokemon-tcg-ai-battle-challenge-strategy` / `EN_Card_Data.csv`.

**Smoke test result:** `libcg.so` (x86_64) loads cleanly on this machine.
`all_card_data()` → 1267, `all_attack()` → 1556 (exact match to `CLAUDE.md`). A full
random-vs-random game via `battle_start`/`battle_select`/`battle_finish` completed in 48
selections, `result=0`. No engine-portability blocker — the Linux `.so` ships directly in
the Kaggle bundle itself, not only on the Windows machine.

## Local play API (`cg.game`)

```python
def battle_start(deck0: list[int], deck1: list[int]) -> tuple[dict | None, StartData]
def battle_select(select_list: list[int]) -> dict
def battle_finish() -> None
def visualize_data() -> str
```
`StartData` = `{battlePtr, errorPlayer, errorType}`. `battle_ptr` is `None`/`0` on failure —
check before continuing. Loop until `obs["current"]["result"] != -1` (0/1 = winner index,
2 = draw).

## Search API (`cg.api`) — confirmed real, not aspirational

```python
def search_begin(agent_observation: Observation,
                  your_deck: list[int], your_prize: list[int],
                  opponent_deck: list[int], opponent_prize: list[int],
                  opponent_hand: list[int], opponent_active: list[int],
                  manual_coin: bool = False) -> SearchState
def search_step(search_id: int, select: list[int]) -> SearchState
def search_end() -> None
def search_release(search_id: int) -> None
```
This is exactly the determinization interface the project philosophy requires: you supply
a **guessed** full hidden state (opponent's deck order, hand, prize cards, and — only if
their active is face-down — their active Pokémon) alongside the real `agent_observation`
(must be passed through unmodified; it carries `search_begin_input`, an opaque token the
engine needs). The engine validates *counts* match reality (`len(opponent_hand) ==
handCount`, etc.) but not *identity* — so sampling a plausible-but-wrong hidden world is
exactly the intended use, not a bug to work around.

`SearchState = {observation: Observation, searchId: int}`. `ApiResult = {state, error}` —
`search_begin`/`search_step` raise on nonzero `error` (invalid card ID, bad active guess,
size mismatch, released/finished search, etc. — see `cg/api.py` for the full table).

## Observation schema (`cg.api.Observation`, dataclass-mapped from the raw dict)

```
Observation: select: SelectData | None, logs: list[Log], current: State | None,
             search_begin_input: str | None
```
- `select is None` **only** at the very first call (return the 60-card deck then).
- `current is None` iff `select is None` (paired).

```
State: turn, turnActionCount, yourIndex (0/1), firstPlayer, supporterPlayed, stadiumPlayed,
       energyAttached, retreated, result (-1 = ongoing, 0/1 = winner, 2 = draw),
       stadium: list[Card] (0-1 elems), looking: list[Card|None]|None,
       players: list[PlayerState]  # len 2, index by yourIndex / 1-yourIndex
```
```
PlayerState: active: list[Pokemon|None] (0-1 elems), bench: list[Pokemon], benchMax,
             deckCount, discard: list[Card], prize: list[Card|None]  # None = face-down,
             handCount, hand: list[Card] | None   # <-- None for the OPPONENT, confirmed live
             poisoned, burned, asleep, paralyzed, confused
```
**Confirmed by direct observation:** own `hand` is a populated `list[Card]`; opponent's
`hand` field is `None` and only `handCount` is given. `prize` entries are `None` while
face-down (own and opponent's un-drawn prizes both start face-down). This is the exact
information boundary Phase 2 (`GameState`/`parse_obs`) must preserve.

```
Pokemon: id (CardData id), serial (unique per-card-instance in this match), hp, maxHp,
         appearThisTurn, energies: list[EnergyType], energyCards: list[Card],
         tools: list[Card], preEvolution: list[Card]
Card: id, serial, playerIndex
```

## Legal options (`SelectData` / `Option`)

```
SelectData: type: SelectType, context: SelectContext, minCount, maxCount,
            remainDamageCounter, remainEnergyCost, option: list[Option],
            deck: list[Card] | None,      # populated only when selecting from own deck
            contextCard: Card | None, effect: Card | None
```
Pick indices into `option`; return between `minCount` and `maxCount` of them, no dupes —
engine enforces this (raises `IndexError`/`ValueError` via `battle_select`/`search_step`
otherwise). **The agent never invents legality — `option` is always the complete legal
set.**

`Option.type` is an `OptionType` (`PLAY, ATTACH, EVOLVE, ABILITY, DISCARD, RETREAT, ATTACK,
END, CARD, TOOL_CARD, ENERGY_CARD, ENERGY, SKILL, NUMBER, YES, NO, SPECIAL_CONDITION`) and
carries only the fields relevant to that type (`area`/`index`/`playerIndex` for board refs,
`attackId` for attacks, etc. — see `cg/api.py:120-187` for the full per-type field table,
copied verbatim in comments there). `SelectType`/`SelectContext` (`cg/api.py:55-118`)
describe *what kind* of choice this is (e.g. `SETUP_ACTIVE_POKEMON`, `TO_BENCH`, `ATTACK`,
`RETREAT`'s target-Pokémon `SWITCH` context) — this is the classification scheme Phase 2's
option-decoder should key off of.

## Card data join (`cg.api.CardData` / `Attack` ↔ `EN_Card_Data.csv`)

**Join key confirmed exact:** engine `CardData.cardId` == CSV column `Card ID` (int,
stringified in the CSV). Verified on cardId 1 (`Basic {G} Energy`), 721 (`Kyogre`, 2 rows =
2 attacks), 722 (`Snover`, 2 rows), 1158 (`Maximum Belt`, ACE SPEC tool). Multi-attack cards
share one `Card ID` across CSV rows exactly as `CLAUDE.md` describes.

```
CardData: cardId, name, cardType: CardType, retreatCost, hp, weakness: EnergyType|None,
          resistance: EnergyType|None, energyType: EnergyType, basic, stage1, stage2, ex,
          megaEx, tera, aceSpec, evolvesFrom: str|None, skills: list[Skill],
          attacks: list[int]   # attack IDs, look up via all_attack()
Attack: attackId, name, text, damage, energies: list[EnergyType]
```
CSV columns (17): `Card ID, Card Name, Expansion, Collection No., Stage.../Type...,
Rule, Category, Previous stage, HP, Type, Weakness, Resistance (Type), Retreat, Move Name,
Cost, Damage, Effect Explanation`. `n/a`/empty strings appear for non-applicable fields
(e.g. energy cards have `HP: n/a`). CSV holds the natural-language `Effect Explanation`
the engine's `Skill.text`/`Attack.text` also carry in shorter form — CSV is the richer
source for report-writing; engine fields are what the agent should reason over at runtime
(no CSV parsing needed in the hot inference path).

## deck.csv format

Plain text, **60 lines, one `Card ID` (int) per line** — not a headered CSV. Confirmed via
the sample deck (`1158` ACE SPEC tool ×1, `721` Kyogre ×2, `722` Snover ×2, ...).
`main.py`'s `read_deck_csv()` falls back to `/kaggle_simulations/agent/deck.csv` if the
relative path doesn't exist — Phase 5 packaging must preserve this fallback path.

## Not yet exercised (deferred to later phases, not needed for the baseline)
- `search_begin`/`search_step` were read and understood but not yet call-tested end-to-end
  (no MCTS layer exists yet — this is Phase-N+1 work per the project philosophy, not part
  of the foundations plan).
- Effects that trigger `SelectContext` values beyond what a ~60-selection random game
  happened to reach (e.g. `EVOLVE`, `DISCARD_ENERGY_CARD`) — Phase 2's option classifier
  should handle the full enum, not just what one sample game exercised.
