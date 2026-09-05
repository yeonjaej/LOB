"""A fundamental-value oracle driven by a real historical price series, instead of
ABIDES's generic synthetic mean-reverting random walk (SparseMeanRevertingOracle).

ValueAgents anchor their trading to whatever this oracle reports, so using AAPL's
actual LOBSTER mid-price series here makes the background market's drift match the
real day's drift, instead of ABIDES's uncalibrated demo defaults (which produced
session-long drift an order of magnitude larger than AAPL's real ~1.3% move -- see
the Part 3b caveats).
"""

import numpy as np

from abides_markets.oracles.oracle import Oracle


class HistoricalOracle(Oracle):
    """`times_ns`/`prices_cents` are pre-sorted, aligned 1:1 arrays. The value at any
    current_time is looked up via nearest-preceding index (step-held between points),
    so the series doesn't need to cover every nanosecond.

    NOTE: `current_time` is always a raw NanosecondTime int in this codebase, never a
    pandas Timestamp -- an easy mistake to make (observed in a sibling notebook's
    attempt at this same oracle, which called `current_time.floor("S")` and would
    have raised AttributeError the moment any ValueAgent asked for a price).
    """

    def __init__(self, times_ns, prices_cents):
        self.times_ns = np.asarray(times_ns, dtype="int64")
        self.prices_cents = np.asarray(prices_cents, dtype="int64")
        assert len(self.times_ns) == len(self.prices_cents)
        assert np.all(np.diff(self.times_ns) >= 0), "times_ns must be sorted ascending"

    def _price_at(self, current_time) -> int:
        idx = np.searchsorted(self.times_ns, current_time, side="right") - 1
        idx = int(np.clip(idx, 0, len(self.prices_cents) - 1))
        return int(self.prices_cents[idx])

    def get_daily_open_price(self, symbol: str, mkt_open, cents: bool = True) -> int:
        return self._price_at(mkt_open)

    def observe_price(self, symbol: str, current_time, random_state, sigma_n: int = 0) -> int:
        r_t = self._price_at(current_time)
        if sigma_n > 0 and random_state is not None:
            r_t += int(round(random_state.normal(0, np.sqrt(sigma_n))))
        return max(r_t, 1)
