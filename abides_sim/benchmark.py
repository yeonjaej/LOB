"""K-seed paired benchmark: naive / TWAP / POV / PPO (no oracle) / PPO (with oracle),
each measured against ONE shared zero-execution baseline per seed (all candidates for
a seed face bit-identical pre-trade background-market state, since every run below is
built from scratch via `build_market_config(seed=...)` -- never a reused/deepcopied
Kernel), followed by paired significance testing against TWAP -- the same
`ttest_rel`/`wilcoxon` pattern already validated in `LOBdata_EDA.ipynb` Part 4 cell 44.

Seed hygiene: BENCHMARK_SEEDS must stay disjoint from the seeds used to train the
oracle (`oracle_train.train_oracle` used 300-349) -- otherwise the oracle's benchmark
performance would be inflated by evaluating it on its own training data.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from abides_core import abides
from abides_core.utils import str_to_ns, datetime_str_to_ns

from .config import build_market_config
from .agents import ScheduledExecutionAgent
from .gym_env import ABIDESExecutionEnv, pick_window
from .run import mid_price_path, mid_at

BENCHMARK_SEEDS = list(range(1, 31))  # disjoint from oracle_train's seeds 300-349
HISTORICAL_DATE_NS = datetime_str_to_ns("20250101")


def _fresh_run(seed, start_time, end_time, exec_agent_builder, symbol="ABM", **market_kwargs):
    """Builds config + Kernel entirely from scratch every call -- never reuses or
    deepcopies a partially-run Kernel -- so a given seed guarantees bit-identical
    pre-trade background state across every candidate evaluated against it."""
    config = build_market_config(
        seed=seed, symbol=symbol, start_time=start_time, end_time=end_time,
        exec_agent_builder=exec_agent_builder, **market_kwargs,
    )
    end_state = abides.run(config)
    return config, end_state


def run_baseline(seed: int, window_minutes: int = 45, symbol: str = "ABM", **market_kwargs) -> dict:
    start_time, end_time = pick_window(seed, window_minutes)
    mkt_open_ns = HISTORICAL_DATE_NS + str_to_ns(start_time)
    mkt_close_ns = HISTORICAL_DATE_NS + str_to_ns(end_time)
    config, end_state = _fresh_run(seed, start_time, end_time, None, symbol=symbol, **market_kwargs)
    mid_df = mid_price_path(end_state["agents"][0].order_books[symbol], symbol)
    return {
        "start_time": start_time,
        "end_time": end_time,
        "mkt_open_ns": mkt_open_ns,
        "mkt_close_ns": mkt_close_ns,
        "arrival_mid": mid_at(mid_df, mkt_open_ns),
        "final_mid": mid_at(mid_df, mkt_close_ns),
    }


def _summarize(fills, exec_mid, baseline: dict, total_shares: int) -> dict:
    fills_df = pd.DataFrame(fills, columns=["time", "price", "qty"])
    total_filled = int(fills_df["qty"].sum()) if len(fills_df) else 0
    total_cost = float((fills_df["price"] * fills_df["qty"]).sum()) if len(fills_df) else 0.0
    avg_price = (total_cost / total_filled / 100.0) if total_filled else float("nan")  # cents -> dollars

    arrival = baseline["arrival_mid"] / 100.0
    is_bps = (avg_price - arrival) / arrival * 10_000 if total_filled else float("nan")

    final_exec_mid = mid_at(exec_mid, baseline["mkt_close_ns"]) / 100.0
    final_clean_mid = baseline["final_mid"] / 100.0
    impact_bps = (final_exec_mid - final_clean_mid) / final_clean_mid * 10_000

    return {"IS_bps": is_bps, "impact_bps": impact_bps, "shares_filled": total_filled}


def run_scheduled(
    seed: int, mode: str, baseline: dict, total_shares: int = 10_000, n_slices: int = 20,
    symbol: str = "ABM", **market_kwargs,
) -> dict:
    assert mode in ("naive", "twap", "pov")
    start_time, end_time = baseline["start_time"], baseline["end_time"]
    # A NoiseAgent whose assigned wakeup_time falls before mkt_open doesn't actually
    # fire early: TradingAgent's own bootstrap (ask the exchange for market hours on
    # first wakeup, then reschedule to mkt_open + a ~0-100ns offset once the reply
    # arrives) forces it to fire AT mkt_open regardless. So a burst of noise agents
    # lands in the same instant as a trade scheduled at literal mkt_open, and whether
    # the exec agent's order sees them depends on event-queue tie-breaking -- this is
    # what caused naive to occasionally get 0 fills. A short delay before the first
    # trade sidesteps the race entirely (mirrors the 30-min warmup used for full-day
    # sessions in run.py, just much shorter since this window is itself much shorter).
    trade_start_ns = baseline["mkt_open_ns"] + str_to_ns("2min")

    def build_exec_agent(agent_id):
        if mode in ("naive", "twap"):
            n = 1 if mode == "naive" else n_slices
            # np.linspace(a, b, n) returns n points INCLUSIVE of both endpoints --
            # n-1 intervals, with the last trade landing exactly on b. That's a
            # fencepost error: n trades at the start of n equal regions need n+1
            # boundary points with the last one dropped. (The static-replay
            # simulate_twap_buy in LOBdata_EDA.ipynb worked around the same issue
            # with an ad hoc `end_sec - 1`; this is the exact fix instead of an
            # epsilon.) Landing a trade on the literal window-close boundary was
            # silently losing an entire slice's fill -- discovered via the K=30
            # benchmark's naive/TWAP shares_filled coming in short.
            schedule = np.linspace(
                trade_start_ns, baseline["mkt_close_ns"], n + 1
            )[:-1].astype("int64").tolist()
            return ScheduledExecutionAgent(
                id=agent_id, name="EXEC_AGENT", type="ExecutionAgent", symbol=symbol,
                starting_cash=10_000_000, mode=mode, total_shares=total_shares,
                schedule=schedule, log_orders=True,
            )
        else:  # pov
            return ScheduledExecutionAgent(
                id=agent_id, name="EXEC_AGENT", type="ExecutionAgent", symbol=symbol,
                starting_cash=10_000_000, mode="pov", total_shares=total_shares,
                pov_frac=0.1, pov_wake_freq_ns=str_to_ns("2min"), pov_lookback="2min",
                start_time=trade_start_ns, end_time=baseline["mkt_close_ns"],
                log_orders=True,
            )

    config, end_state = _fresh_run(seed, start_time, end_time, build_exec_agent, symbol=symbol, **market_kwargs)
    exec_agent = config["agents"][-1]
    exec_mid = mid_price_path(end_state["agents"][0].order_books[symbol], symbol)
    return _summarize(exec_agent.fills, exec_mid, baseline, total_shares)


def run_ppo(
    seed: int, model, oracle_predict_fn: Optional[callable], baseline: dict,
    total_shares: int = 10_000, horizon_steps: int = 40, window_minutes: int = 45,
    step_interval: str = "1min", symbol: str = "ABM", phi: float = 0.03, psi: float = 1.0,
    **market_kwargs,
) -> dict:
    """Evaluates a trained PPO policy deterministically. Hyperparameters
    (horizon_steps/window_minutes/step_interval/total_shares/phi/psi) must match
    training -- these defaults match what `train_ppo_agent`'s default
    `ABIDESExecutionEnv` used. phi/psi are ABIDESExecutionEnv's own reward-shaping
    constructor args, NOT market_kwargs -- they used to silently fall into
    **market_kwargs here (which only reaches build_market_config, never the env's
    reward), so overriding them had no effect and any mismatched checkpoint was
    silently evaluated under the wrong reward shaping. Now explicit params instead.

    horizon_steps=40 (not 20) so PPO's own execution window (horizon_steps *
    step_interval = 40min, starting at mkt_open) matches run_scheduled's TWAP/naive/
    POV window (mkt_open+2min to mkt_close, ~41-43min for a 45min session) instead of
    giving PPO less than half the time to execute the same total_shares. A blanket
    `run_benchmark(..., horizon_steps=...)` kwarg can't fix this: it flows through
    market_kwargs to run_scheduled too, whose build_market_config() call doesn't
    accept horizon_steps and would raise a TypeError -- this default has to change
    here instead, matched to whatever horizon_steps the current PPO checkpoint was
    actually trained with.

    NOTE: extending the horizon alone (still at phi=0.1) did NOT fix the elevated
    impact_bps -- no-oracle's test-time pace stayed at ~20min even with 40 available,
    unchanged from when horizon_steps=20 forced it. That ruled out "forced by a short
    deadline" as the actual cause; the real driver was the running per-step inventory-
    urgency term `r_inv = -phi * (remaining_frac)**2` in RLExecutionAgent.wakeup(),
    which makes fast unwinding a genuine reward-optimal choice regardless of how much
    time is available (an Almgren-Chriss-style impact/timing-risk tradeoff, correctly
    "learned," not a bug). phi=0.03 (see below) is the actual fix for impact_bps;
    horizon_steps=40 remains necessary so PPO *can* trade at TWAP's pace if the reward
    now makes that worthwhile, not because it *forces* PPO to.

    phi=0.03 (not the environment's own 0.1 default) matches the checkpoints selected
    for `data/ppo_{no,with}_oracle_final.zip`: retrained for 1000 episodes (vs. the
    original 500) at the lower urgency weight, with periodic snapshots every 100
    episodes, and the best snapshot picked by held-out validation-seed impact_bps
    rather than just keeping the final one -- ep_len_mean (and impact_bps) drifted
    back down late in with-oracle's training, worse than its own mid-training peak, so
    trusting the final checkpoint would have picked a worse policy. This brought
    impact_bps down to ~TWAP-level (from ~10-16bps at phi=0.1 down to roughly 0bps).
    """
    env = ABIDESExecutionEnv(
        total_shares=total_shares, horizon_steps=horizon_steps, window_minutes=window_minutes,
        step_interval=step_interval, oracle_predict_fn=oracle_predict_fn, phi=phi, psi=psi,
        market_kwargs=dict(symbol=symbol, **market_kwargs),
    )
    obs, _ = env.reset(seed=seed)
    terminated = False
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

    exec_mid = mid_price_path(env.kernel.agents[0].order_books[symbol], symbol)
    return _summarize(env.gym_agent.fills, exec_mid, baseline, total_shares)


def _run_one_seed(
    seed: int, model_no_oracle_path: str, model_with_oracle_path: str,
    oracle_predict_fn: Optional[callable], market_kwargs: dict,
) -> list:
    """One seed's full bundle (baseline + all 5 candidates) -- the unit of work handed
    to each ProcessPoolExecutor worker. Loads the PPO models from disk inside the
    worker rather than receiving live model objects, sidestepping any pickling quirks
    around SB3's internal device/optimizer state when passed across a process
    boundary; loading fresh per-worker is cheap relative to the simulation cost it
    runs alongside. oracle_predict_fn (an `OraclePredictor` instance) IS passed
    directly -- it was deliberately built as a plain, picklable, module-level class."""
    from stable_baselines3 import PPO

    model_no_oracle = PPO.load(model_no_oracle_path)
    model_with_oracle = PPO.load(model_with_oracle_path)
    baseline = run_baseline(seed, **market_kwargs)  # computed once, shared below
    candidates = {
        "Naive": run_scheduled(seed, "naive", baseline, **market_kwargs),
        "TWAP": run_scheduled(seed, "twap", baseline, **market_kwargs),
        "POV": run_scheduled(seed, "pov", baseline, **market_kwargs),
        "PPO (no oracle)": run_ppo(seed, model_no_oracle, None, baseline, **market_kwargs),
        "PPO (with oracle)": run_ppo(seed, model_with_oracle, oracle_predict_fn, baseline, **market_kwargs),
    }
    print(f"seed {seed} done: " + ", ".join(f"{k}={v['IS_bps']:+.1f}bps" for k, v in candidates.items()))
    return [{"seed": seed, "method": label, **res} for label, res in candidates.items()]


def run_benchmark(
    model_no_oracle_path: str, model_with_oracle_path: str, oracle_predict_fn,
    seeds=BENCHMARK_SEEDS, n_workers: Optional[int] = None, **market_kwargs,
) -> pd.DataFrame:
    """Each of the 30 (by default) seeds is an independent full-window bundle (one
    shared baseline run + 5 candidate runs), so seeds are farmed out across a process
    pool -- same "trivially parallel across independent ABIDES simulations" pattern
    already used for the oracle's data generation.

    NOTE: takes model *paths* (strings), not live PPO objects -- see `_run_one_seed`.
    """
    assert set(seeds).isdisjoint(range(300, 350)), "benchmark seeds overlap the oracle's training seeds"
    n_workers = min(n_workers or max(1, (os.cpu_count() or 2) - 1), len(seeds))
    worker_fn = partial(
        _run_one_seed, model_no_oracle_path=model_no_oracle_path,
        model_with_oracle_path=model_with_oracle_path,
        oracle_predict_fn=oracle_predict_fn, market_kwargs=market_kwargs,
    )
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(worker_fn, seeds))
    rows = [row for seed_rows in results for row in seed_rows]
    return pd.DataFrame(rows)


def paired_tests(df: pd.DataFrame, baseline_method: str = "TWAP", metric: str = "IS_bps") -> pd.DataFrame:
    base = df[df["method"] == baseline_method].set_index("seed")[metric]
    results = {}
    for method in df["method"].unique():
        if method == baseline_method:
            continue
        cand = df[df["method"] == method].set_index("seed")[metric]
        common = base.index.intersection(cand.index)
        b, c = base.loc[common].values, cand.loc[common].values
        d = c - b
        t_stat, p_t = stats.ttest_rel(c, b)
        w_stat, p_w = stats.wilcoxon(c, b)
        results[method] = {
            "mean_diff": d.mean(),
            "std_diff": d.std(ddof=1),
            "sem": d.std(ddof=1) / np.sqrt(len(d)),
            "t_stat": t_stat,
            "p_ttest": p_t,
            "w_stat": w_stat,
            "p_wilcoxon": p_w,
        }
    return pd.DataFrame(results).T
