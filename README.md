# RL for Optimal Trade Execution in a Simulated Limit Order Book

**Work in progress.** This is an ongoing research project, not a finished result or a
library — expect rough edges, and read the "Current status" and "Known limitations"
sections before taking any number here at face value.

## What this is

The problem is *optimal execution*: buy a fixed quantity (10,000 shares) over a fixed
window (~40 minutes) at the best achievable average price. Trading too fast walks the
order book and moves the price against you; trading too slowly exposes you to price
drift and risks not finishing. Classic solutions are scheduled: slice the order evenly
over time (TWAP), or track market volume (POV).

This project asks whether a reinforcement-learning agent can do better by *reacting* to
live market conditions. A PPO policy is trained inside a running agent-based market
simulation and benchmarked against naive / TWAP / POV baselines on identical market
conditions, using paired statistical tests across 30 random seeds.

## Current status

The learned policy **performs on par with TWAP — it does not beat it.** It does clearly
beat the naive and POV baselines.

K=30 paired benchmark, implementation shortfall in basis points (lower is better):

| method | IS (bps) | market impact (bps) |
|---|---|---|
| TWAP | **−1.00** | 7.83 |
| PPO (learned) | **−0.44** | 8.83 |
| POV | +3.71 | 8.45 |
| Naive (single order) | +9.00 | 7.60 |

PPO − TWAP = **+0.56 bps, 95% CI [−5.24, +6.35], p = 0.85** — statistically
indistinguishable from TWAP. That confidence interval is wide for a reason: with 30
seeds the benchmark's minimum detectable effect is about 7.75 bps, while the
differences between design variants are 1–5 bps. **The evaluation cannot resolve the
effects being chased**, which is itself one of the project's more useful findings, and
the reason later work leaned on low-variance diagnostics rather than the benchmark.

A plausible explanation for the parity: at ~3% participation over a compressed window,
spreading the order evenly is close to optimal, so there is little edge available. The
trained policy in fact converges toward TWAP-like behaviour on its own — it trades a
mean of ~241 shares per step against the 250-share uniform slice.

## Background: ABIDES

[ABIDES](https://github.com/jpmorganchase/abides-jpmc-public) (JPMorgan Chase) is an
open-source agent-based, discrete-event market simulator. Rather than replaying
historical data, it runs a population of background agents — noise traders, value
traders, momentum traders, market makers — whose interactions produce an *emergent*
limit order book with realistic microstructure.

The key property for this project: you can inject your own trading agent, and the
market **reacts to its orders**. An execution agent's own trades consume liquidity and
move the price, so market impact is simulated endogenously rather than assumed from a
model. Static historical replay cannot capture that feedback.

## Repo layout

```
abides_sim/          # reusable simulation + training code
  config.py          # trimmed RMSC-3-style background market (exchange, noise,
                     #   value, market-maker, momentum agents) + run_market()
  agents.py          # scheduled execution agents: naive / TWAP / POV
  gym_env.py         # Gymnasium env wrapping a *live* ABIDES kernel; the RL agent,
                     #   observation space, action space and reward live here
  train_ppo.py       # PPO training loop with checkpointing + progress logging
  benchmark.py       # K-seed paired benchmark + significance testing
  run.py             # run scheduled strategies and extract comparable metrics
  oracle.py          # fundamental-value oracle driven by a real price series
  oracle_train.py    # trains the optional return-prediction oracle feature

data/                # notebooks (tracked); checkpoints, CSVs, plots (gitignored)
log/                 # ABIDES simulation logs (gitignored)
```

`gym_env.py` is the heart of it — the bridge between Gymnasium's synchronous
`step()`/`reset()` API and ABIDES's blocking discrete-event loop, plus every
observation/action/reward design decision.

Note that **only source and notebooks are tracked**. Trained checkpoints, benchmark
CSVs and logs are gitignored, so results need regenerating from scratch.

## Setup

Requires Python 3.14, `stable_baselines3` 2.9, `gymnasium` 1.3, `torch` 2.13,
plus `pandas` / `numpy` / `scipy` / `scikit-learn`.

ABIDES itself is **not vendored**. Clone it as a sibling directory:

```
Projects/
├── abides-jpmc-public/     # git clone from jpmorganchase/abides-jpmc-public
└── LOB/                    # this repo
```

> **Portability caveat — you will hit this first.** `abides_sim/__init__.py` inserts
> two *hardcoded absolute paths* to `abides-jpmc-public/abides-core` and
> `abides-markets` on the original author's machine. If you clone this anywhere else,
> edit those two paths (or replace them with a relative/`site-packages` install)
> before anything will import.

### Training a policy

```python
from abides_sim.train_ppo import train_ppo_agent

train_ppo_agent(
    target_episodes=1000,
    checkpoint_path="ppo_no_oracle",
    env_kwargs=dict(
        total_shares=10_000, horizon_steps=40, window_minutes=45,
        step_interval="1min", continuous_action=True,
        phi=0.03, psi=1.0, market_kwargs=dict(symbol="ABM"),
    ),
    n_envs=8, snapshot_every=100, gamma=1.0,
)
```

One episode is a full ~45-minute market simulation, so training is CPU-bound and slow:
1000 episodes takes roughly 70–100 minutes on 8 parallel environments.

### Benchmarking

```python
from abides_sim.benchmark import run_benchmark, paired_tests

df = run_benchmark("ppo_no_oracle", "ppo_with_oracle", oracle_predict_fn=None)
print(paired_tests(df, baseline_method="TWAP", metric="IS_bps"))
```

Each of the 30 seeds runs every candidate against a bit-identical background market, so
comparisons are properly paired.

## Notebooks — where the reasoning lives

The design decisions, dead ends and corrections are documented in the notebooks rather
than here. Each cell's output is genuinely executed, not transcribed.

| notebook | contents |
|---|---|
| `LOBdata_EDA.ipynb` | LOBSTER AAPL order-book EDA, and the original progression from static replay through to live-ABIDES RL |
| `LOBdata_RL_v1.ipynb` | richer observation space |
| `LOBdata_RL_v2.ipynb` | finer discrete action space, and an exploration-pressure sweep |
| `LOBdata_RL_v3.ipynb` | continuous action space; reward design, market-impact attribution, and observation-feature analysis |

`LOBdata_RL_v3.ipynb` is the most developed and the best starting point for the
methodology — including several results that overturned earlier conclusions.

## Known limitations

Three things to know before reading any result here as meaningful:

1. **The horizon is compressed.** 10,000 shares in ~40 minutes is roughly **3%** of
   simulated volume — an urgent liquidation, not a realistic full-day execution
   mandate (which would be ~0.3–0.4%). The tradeoffs, and how much edge is available,
   differ substantially between those two regimes.
2. **The simulated book is thin.** Price level and volatility are calibrated to AAPL,
   but depth is not: median resting depth at the touch is ~48 shares, against a real
   large-cap's far deeper book. Absolute impact and slippage numbers are therefore
   inflated relative to a real venue.
3. **The benchmark is underpowered.** ~7.75 bps minimum detectable effect at n=30
   versus 1–5 bps effects. Version-to-version comparisons in this project are mostly
   *not* statistically resolvable, and are reported with confidence intervals for
   exactly that reason.

## Possible next steps

- Extend to a full-trading-day horizon at realistic participation, where timing
  decisions plausibly have more room to matter.
- Raise benchmark power (more seeds, or variance reduction) enough to resolve the
  effect sizes involved.
- Act on the observation-feature analysis in `LOBdata_RL_v3.ipynb` §15 — notably that
  bid-ask spread carries value-relevant signal the current policy largely ignores.
