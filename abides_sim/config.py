"""Trimmed RMSC-3-style ABIDES background market (Exchange + Noise + Value +
AdaptiveMarketMaker + Momentum agents), avoiding the upstream rmsc03.py config
directly since it imports a POVExecutionAgent that doesn't exist in the public repo.

The execution agent (if any) is injected via `exec_agent_builder` so the exact same
background-market code path produces both a "baseline" (no execution agent) and a
"with execution agent" run. To keep the background agents' realized behavior identical
between a baseline run and an execution run for the same seed (so any difference in the
resulting price path is attributable to the execution agent's own trading, i.e. market
impact, not to incidental RNG-stream drift), the latency model and kernel random state
are always sized for `base_agent_count + 1` — one slot reserved for the execution agent
whether or not it's actually present. This keeps every `np.random` draw consumed while
building the background market identical across baseline/naive/twap/pov runs of the
same seed.
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd

from abides_core.utils import str_to_ns, datetime_str_to_ns, get_wake_time
from abides_markets.agents import (
    ExchangeAgent,
    NoiseAgent,
    ValueAgent,
    AdaptiveMarketMakerAgent,
    MomentumAgent,
)
from abides_markets.oracles import SparseMeanRevertingOracle
from abides_markets.utils import generate_latency_model

from .oracle import HistoricalOracle


def build_market_config(
    seed: int,
    symbol: str = "ABM",
    historical_date_str: str = "20250101",
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
    fund_r_bar: int = 58_500,  # fundamental value in CENTS; ~$585.00, AAPL's Part-1 level
    fund_vol: float = 1.2e-4,  # see docstring: recalibrated from rmsc03.py's 1e-3
    fundamental_series: Optional[pd.Series] = None,
    num_noise_agents: int = 5000,
    num_value_agents: int = 100,
    num_momentum_agents: int = 25,
    num_mm_agents: int = 2,
    log_orders: bool = True,
    book_log_depth: int = 10,
    noise_warmup: str = "30min",
    exec_agent_builder: Optional[Callable[[int], object]] = None,
) -> dict:
    """`fundamental_series`, if given, replaces ABIDES's generic synthetic
    mean-reverting oracle with a real historical one: a pandas Series indexed by
    time-of-day in seconds (e.g. Part 1's `df["time_sec"]`), values = mid-price in
    dollars (e.g. Part 1's `df["mid_price"]`). ValueAgents then anchor to AAPL's
    actual price path -- including its real, modest drift -- instead of a synthetic
    random walk whose volatility/megashock parameters were never calibrated to AAPL
    (see the Part 3b caveats: that mismatch is what produced session-long drift an
    order of magnitude larger than AAPL's real ~1.3% move).

    `fund_vol` (only used on the synthetic path, i.e. when `fundamental_series` is
    None): the OU fundamental process's diffusion variance scales as
    `fund_vol**2 * elapsed_ns`. At `rmsc03.py`'s original 1e-3 (never recalibrated
    for this project's price level/session length), a 45-min window's fundamental
    alone has std ~$16.43 -- a ~2.8% swing from diffusion before any order-flow
    amplification, versus Part 1's real AAPL day moving only ~1.3% over the *entire*
    6.5-hour session. 1.2e-4 targets ~$1.75-$2.93 (0.3-0.5%) of fundamental-only
    movement over a 45-min window, scaling Part 1's realized move by sqrt(45/390).
    This was the dominant driver of Part 10's K=30 noise floor and several outlier
    seeds (e.g. +284, +358 bps) -- fixing the reward function's drift-sensitivity
    was fighting a symptom of this, not the cause.
    """
    np.random.seed(seed)

    historical_date = datetime_str_to_ns(historical_date_str)
    mkt_open = historical_date + str_to_ns(start_time)
    mkt_close = historical_date + str_to_ns(end_time)
    starting_cash = 10_000_000

    if fundamental_series is not None:
        times_ns = historical_date + (fundamental_series.index.values * 1e9).astype("int64")
        prices_cents = np.round(fundamental_series.values * 100).astype("int64")
        oracle = HistoricalOracle(times_ns, prices_cents)
        fund_r_bar = int(prices_cents[0])
    else:
        symbols = {
            symbol: {
                "r_bar": fund_r_bar,
                "kappa": 1.67e-16,
                "sigma_s": 0,
                "fund_vol": fund_vol,
                "megashock_lambda_a": 2.77778e-18,
                "megashock_mean": 1000,
                "megashock_var": 50_000,
                "random_state": np.random.RandomState(
                    seed=np.random.randint(low=0, high=2**32, dtype="uint64")
                ),
            }
        }
        oracle = SparseMeanRevertingOracle(mkt_open, mkt_close, symbols)

    fund_sigma_n = fund_r_bar / 10

    agents = [
        ExchangeAgent(
            id=0,
            name="EXCHANGE_AGENT",
            mkt_open=mkt_open,
            mkt_close=mkt_close,
            symbols=[symbol],
            book_logging=True,
            book_log_depth=book_log_depth,
            log_orders=log_orders,
            pipeline_delay=0,
            computation_delay=0,
            stream_history=25_000,
        )
    ]
    agent_count = 1

    # NOTE: previously hardcoded to "09:00:00"-"16:00:00" regardless of start_time/
    # end_time -- harmless for a full 6.5-hour session, but for a short (e.g. 30-60
    # min) training episode it meant only a tiny fraction of num_noise_agents ever
    # woke up inside the window, giving unrealistically thin liquidity. Scale to the
    # actual session instead.
    noise_mkt_open = mkt_open - str_to_ns(noise_warmup)
    noise_mkt_close = mkt_close
    agents += [
        NoiseAgent(
            id=j,
            symbol=symbol,
            starting_cash=starting_cash,
            wakeup_time=get_wake_time(noise_mkt_open, noise_mkt_close),
            log_orders=log_orders,
        )
        for j in range(agent_count, agent_count + num_noise_agents)
    ]
    agent_count += num_noise_agents

    agents += [
        ValueAgent(
            id=j,
            name=f"Value Agent {j}",
            symbol=symbol,
            starting_cash=starting_cash,
            sigma_n=fund_sigma_n,
            r_bar=fund_r_bar,
            kappa=1.67e-15,
            lambda_a=7e-11,
            log_orders=log_orders,
        )
        for j in range(agent_count, agent_count + num_value_agents)
    ]
    agent_count += num_value_agents

    agents += [
        AdaptiveMarketMakerAgent(
            id=j,
            name=f"ADAPTIVE_POV_MARKET_MAKER_AGENT_{j}",
            type="AdaptivePOVMarketMakerAgent",
            symbol=symbol,
            starting_cash=starting_cash,
            pov=0.025,
            min_order_size=1,
            window_size="adaptive",
            num_ticks=10,
            wake_up_freq=str_to_ns("10S"),
            cancel_limit_delay=50,
            skew_beta=0,
            level_spacing=5,
            spread_alpha=0.75,
            backstop_quantity=50_000,
            log_orders=log_orders,
        )
        for j in range(agent_count, agent_count + num_mm_agents)
    ]
    agent_count += num_mm_agents

    agents += [
        MomentumAgent(
            id=j,
            name=f"MOMENTUM_AGENT_{j}",
            symbol=symbol,
            starting_cash=starting_cash,
            min_size=1,
            max_size=10,
            wake_up_freq=str_to_ns("20s"),
            log_orders=log_orders,
        )
        for j in range(agent_count, agent_count + num_momentum_agents)
    ]
    agent_count += num_momentum_agents

    base_agent_count = agent_count

    # Always size the latency model / kernel RNG for one extra (execution) agent, whether
    # or not one is actually appended below -- see module docstring.
    latency_model = generate_latency_model(base_agent_count + 1)
    random_state_kernel = np.random.RandomState(
        seed=np.random.randint(low=0, high=2**32, dtype="uint64")
    )

    if exec_agent_builder is not None:
        agents.append(exec_agent_builder(base_agent_count))

    return {
        "start_time": historical_date,
        "stop_time": mkt_close + str_to_ns("00:01:00"),
        "agents": agents,
        "agent_latency_model": latency_model,
        "default_computation_delay": 50,
        "custom_properties": {"oracle": oracle},
        "random_state_kernel": random_state_kernel,
        "stdout_log_level": "WARNING",
        # exposed for run.py's metric computation
        "symbol": symbol,
        "mkt_open": mkt_open,
        "mkt_close": mkt_close,
    }
