"""Run naive/TWAP/POV execution strategies inside a live ABIDES market and extract
metrics shaped like the notebook's existing `simulate_scheduled_execution` output, so
Part 2/3's plotting cells can be reused with minimal changes.

Each call runs TWO simulations for the same seed: a baseline (no execution agent) and
one with the execution agent appended. Because `config.build_market_config` always
reserves the execution agent's RNG "slot" (see its docstring), the background market
realization is identical between the two runs up to the point the execution agent's own
orders start perturbing it -- so the difference between the two mid-price paths is
attributable to the execution agent's trading, i.e. market impact.
"""

from typing import Optional

import numpy as np
import pandas as pd

from abides_core import abides
from abides_core.utils import str_to_ns

from .agents import ScheduledExecutionAgent
from .config import build_market_config


def _side_series(l1_side: np.ndarray) -> pd.Series:
    """L1 best_bids/best_asks array -> price series indexed by time, NaNs dropped
    (ABIDES only appends a row on that side when it actually changes)."""
    df = pd.DataFrame(l1_side, columns=["time", "price", "qty"])
    df = df.dropna(subset=["price"])
    df["time"] = df["time"].astype("int64")
    df["price"] = df["price"].astype(float)
    return df.set_index("time")["price"].sort_index()


def mid_price_path(order_book, symbol: str) -> pd.DataFrame:
    L1 = order_book.get_L1_snapshots()
    bids = _side_series(L1["best_bids"])
    asks = _side_series(L1["best_asks"])
    times = np.union1d(bids.index.values, asks.index.values).astype("int64")
    out = pd.DataFrame({"time": times})
    out = pd.merge_asof(out, bids.rename("bid").reset_index(), on="time", direction="backward")
    out = pd.merge_asof(out, asks.rename("ask").reset_index(), on="time", direction="backward")
    out["mid"] = (out["bid"] + out["ask"]) / 2.0
    return out.dropna(subset=["mid"]).reset_index(drop=True)


def mid_at_many(mid_df: pd.DataFrame, times) -> np.ndarray:
    # "nearest" rather than "backward" so a query at/near market open (before the book's
    # first logged quote) still resolves to that first quote instead of NaN. Vectorized:
    # one merge_asof call for all query times rather than one per point.
    query = pd.DataFrame({"time": np.asarray(times, dtype="int64")})
    order = np.argsort(query["time"].values, kind="stable")
    sorted_query = query.iloc[order].reset_index(drop=True)
    merged = pd.merge_asof(sorted_query, mid_df[["time", "mid"]], on="time", direction="nearest")
    result = np.empty(len(times))
    result[order] = merged["mid"].values
    return result


def mid_at(mid_df: pd.DataFrame, t: int) -> float:
    return float(mid_at_many(mid_df, [t])[0])


def run_execution_sim(
    mode: str,
    seed: int,
    total_shares: int = 10_000,
    n_slices: int = 20,
    pov_frac: float = 0.1,
    pov_wake_freq: str = "1min",
    symbol: str = "ABM",
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
    fund_r_bar: int = 58_500,
    fundamental_series: Optional[pd.Series] = None,
    num_noise_agents: int = 5000,
    num_value_agents: int = 100,
    num_momentum_agents: int = 25,
    label: Optional[str] = None,
):
    """Returns (summary: dict, trades_df: pd.DataFrame, baseline_mid: df, exec_mid: df).

    `fundamental_series`, if given, is passed through to `build_market_config` --
    a real historical price series (e.g. Part 1's mid-price) driving the background
    market's fundamental value instead of ABIDES's generic synthetic oracle.
    """

    assert mode in ("naive", "twap", "pov")
    label = label or {"naive": "Naive single-shot", "twap": "TWAP (time-clock)", "pov": "POV (volume-clock)"}[mode]

    common_kwargs = dict(
        symbol=symbol,
        start_time=start_time,
        end_time=end_time,
        fund_r_bar=fund_r_bar,
        fundamental_series=fundamental_series,
        num_noise_agents=num_noise_agents,
        num_value_agents=num_value_agents,
        num_momentum_agents=num_momentum_agents,
    )

    # --- baseline: identical background market, no execution agent ---
    baseline_config = build_market_config(seed=seed, exec_agent_builder=None, **common_kwargs)
    baseline_end_state = abides.run(baseline_config)
    baseline_mid = mid_price_path(baseline_end_state["agents"][0].order_books[symbol], symbol)

    mkt_open_ns = baseline_config["mkt_open"]
    mkt_close_ns = baseline_config["mkt_close"]
    # "Arrival" (parent-order decision time) is still literal market open, but actual
    # child-order trading starts 30 min after open / ends 30 min before close -- the
    # book has zero resting liquidity at the literal instant of open (nothing has had a
    # chance to quote yet), and this is the same warm-up abides-jpmc-public's own
    # rmsc03.py reference config uses for its execution agent.
    warmup = str_to_ns("30min")
    trade_start_ns = mkt_open_ns + warmup
    trade_end_ns = mkt_close_ns - warmup
    arrival_price = mid_at(baseline_mid, mkt_open_ns)

    # --- execution run: same seed, execution agent appended ---
    def build_exec_agent(agent_id: int):
        if mode in ("naive", "twap"):
            n = 1 if mode == "naive" else n_slices
            schedule = np.linspace(trade_start_ns, trade_end_ns, n).astype("int64").tolist()
            return ScheduledExecutionAgent(
                id=agent_id,
                name="EXECUTION_AGENT",
                type="ExecutionAgent",
                symbol=symbol,
                starting_cash=10_000_000,
                mode=mode,
                total_shares=total_shares,
                schedule=schedule,
                log_orders=True,
            )
        else:
            return ScheduledExecutionAgent(
                id=agent_id,
                name="EXECUTION_AGENT",
                type="ExecutionAgent",
                symbol=symbol,
                starting_cash=10_000_000,
                mode="pov",
                total_shares=total_shares,
                pov_frac=pov_frac,
                pov_wake_freq_ns=str_to_ns(pov_wake_freq),
                pov_lookback=pov_wake_freq,
                start_time=trade_start_ns,
                end_time=trade_end_ns,
                log_orders=True,
            )

    exec_config = build_market_config(seed=seed, exec_agent_builder=build_exec_agent, **common_kwargs)
    exec_end_state = abides.run(exec_config)
    exec_agent = exec_config["agents"][-1]  # same object instance abides.run mutated in place
    exec_mid = mid_price_path(exec_end_state["agents"][0].order_books[symbol], symbol)

    # --- bucket individual fills back into their originating slice ---
    fills = pd.DataFrame(exec_agent.fills, columns=["time", "price", "qty"])
    slices = pd.DataFrame(exec_agent.slice_log, columns=["time", "requested"])
    slices = slices.sort_values("time").reset_index(drop=True)
    slices["slice_num"] = np.arange(1, len(slices) + 1)

    if len(fills) and len(slices):
        fills = fills.sort_values("time")
        fills["slice_num"] = np.searchsorted(slices["time"].values, fills["time"].values, side="right")
        fills["slice_num"] = fills["slice_num"].clip(1, len(slices))
    else:
        fills["slice_num"] = []

    records = []
    total_cost = 0.0
    total_filled = 0
    for _, srow in slices.iterrows():
        sf = fills[fills["slice_num"] == srow["slice_num"]]
        shares = int(sf["qty"].sum())
        if shares == 0:
            continue
        avg_price = float((sf["price"] * sf["qty"]).sum() / shares) / 100.0  # cents -> dollars
        # Use the slice's own *actual fill* time, not its scheduled wake time, to look up
        # the contemporaneous mid: network/computation latency means fills land a few ms
        # to a few tens of ms after the wake-up call that triggers them, and "nearest"
        # against the scheduled time can otherwise grab a stale pre-fill snapshot (this is
        # exactly what silently zeroed out naive's market-impact reading below, before the
        # fix -- see Part 2b/3b discussion).
        fill_time = int(sf["time"].min())
        contemporaneous_mid = mid_at(exec_mid, fill_time) / 100.0
        total_cost += avg_price * shares
        total_filled += shares
        records.append(
            {
                "trade_num": int(srow["slice_num"]),
                "time_ns": fill_time,
                "shares": shares,
                "avg_fill_price": avg_price,
                "contemporaneous_mid": contemporaneous_mid,
                "slippage_vs_mid_bps": (avg_price - contemporaneous_mid) / contemporaneous_mid * 10_000,
            }
        )
    trades_df = pd.DataFrame(records)

    arrival_price_dollars = arrival_price / 100.0
    avg_execution_price = total_cost / total_filled if total_filled else float("nan")
    is_dollars = avg_execution_price - arrival_price_dollars
    is_bps = is_dollars / arrival_price_dollars * 10_000

    benchmark_price = np.average(trades_df["contemporaneous_mid"], weights=trades_df["shares"]) if len(trades_df) else float("nan")
    vs_benchmark_dollars = avg_execution_price - benchmark_price
    vs_benchmark_bps = vs_benchmark_dollars / benchmark_price * 10_000

    # --- market impact: exec-run mid vs. baseline mid, same seed, over the exec window ---
    # Anchored to the *actual* first/last fill times, not the scheduled wake times: for a
    # single-shot naive order in particular, the scheduled time is pre-trade (network/
    # computation latency delays the fill by ~10-20ms), so measuring impact "at" the
    # scheduled instant silently grabs the same pre-trade snapshot in both the baseline and
    # execution runs -- a real bug that showed up as an exact 0.00 impact reading for naive
    # regardless of trade size (diagnosed via the raw L1 snapshot timestamps around a fill).
    if len(fills):
        impact_start_ns = int(fills["time"].min())
        impact_end_ns = int(fills["time"].max())
    else:
        impact_start_ns, impact_end_ns = trade_start_ns, trade_end_ns
    grid = np.linspace(impact_start_ns, impact_end_ns, 200).astype("int64")
    exec_path = mid_at_many(exec_mid, grid)
    base_path = mid_at_many(baseline_mid, grid)
    impact_bps_series = (exec_path - base_path) / base_path * 10_000
    impact_bps_avg = float(np.mean(impact_bps_series))
    impact_bps_end = float(impact_bps_series[-1])

    summary = {
        "label": label,
        "total_shares": total_shares,
        "n_trades": len(trades_df),
        "arrival_price": arrival_price_dollars,
        "benchmark_price": benchmark_price,
        "avg_execution_price": avg_execution_price,
        "IS_dollars": is_dollars,
        "IS_bps": is_bps,
        "vs_benchmark_dollars": vs_benchmark_dollars,
        "vs_benchmark_bps": vs_benchmark_bps,
        "impact_bps_avg": impact_bps_avg,
        "impact_bps_end": impact_bps_end,
        "shares_filled": total_filled,
        "seed": seed,
    }

    print(f"--- {label} (ABIDES, seed={seed}): Execution Summary ---")
    print(f"Total shares          : {total_shares:,} filled {total_filled:,} in {len(trades_df)} slice(s)")
    print(f"Arrival mid-price     : ${arrival_price_dollars:.4f}")
    print(f"Avg execution price   : ${avg_execution_price:.4f}")
    print(f"Implementation Shortfall vs arrival : {is_dollars:+.4f} ({is_bps:+.2f} bps)")
    print(f"Slippage vs benchmark                : {vs_benchmark_dollars:+.4f} ({vs_benchmark_bps:+.2f} bps)")
    print(f"Market impact (avg / end of window)  : {impact_bps_avg:+.2f} / {impact_bps_end:+.2f} bps\n")

    return summary, trades_df, baseline_mid, exec_mid
