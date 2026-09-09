"""Custom ABIDES execution agents.

The public abides-jpmc-public repo's reference configs (rmsc03.py) import a
POVExecutionAgent that was never actually open-sourced, so the naive / TWAP / POV
(volume-participation) strategies below are written from scratch against the
abides_markets.agents.trading_agent.TradingAgent base class.

Naive and TWAP need no message round-trip: place_market_order() walks the live book
directly, same as the notebook's static-replay engine. POV needs one round-trip per
wake-up to learn the volume transacted since the last wake (get_transacted_volume /
query_transacted_volume), since real participation-rate algos can't see future volume.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

from abides_core import NanosecondTime
from abides_markets.agents.trading_agent import TradingAgent
from abides_markets.orders import Side


def build_slice_sizes(total_shares: int, n_slices: int) -> List[int]:
    """Equal split with the remainder distributed 1-per-slice to the earliest slices."""
    base, remainder = divmod(total_shares, n_slices)
    return [base + 1 if i < remainder else base for i in range(n_slices)]


class ScheduledExecutionAgent(TradingAgent):
    """One agent, three strategies:

    mode="naive": schedule is a single wake time, one slice = total_shares.
    mode="twap":  schedule is N evenly time-spaced wake times, equal slices.
    mode="pov":   wakes every `pov_wake_freq_ns`, buys `pov_frac` of the volume
                  transacted since its last wake-up (no lookahead).
    """

    def __init__(
        self,
        id: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        symbol: str = "ABM",
        starting_cash: int = 10_000_000,
        log_orders: bool = True,
        mode: str = "twap",
        total_shares: int = 10_000,
        direction: Side = Side.BID,
        schedule: Optional[Sequence[NanosecondTime]] = None,
        pov_frac: Optional[float] = None,
        pov_wake_freq_ns: Optional[int] = None,
        pov_lookback: str = "1min",
        start_time: Optional[NanosecondTime] = None,
        end_time: Optional[NanosecondTime] = None,
    ) -> None:
        super().__init__(id, name, type, random_state, starting_cash, log_orders)

        assert mode in ("naive", "twap", "pov")
        self.mode = mode
        self.symbol = symbol
        self.direction = direction
        self.total_shares = total_shares
        # `remaining` is a derived property (see below), not a mutable field -- see
        # the matching fix/comment on RLExecutionAgent.remaining in gym_env.py for
        # why decrementing at request time (the previous design here too) silently
        # masks partial-fill shortfalls instead of reporting them.
        self.total_filled = 0

        if mode in ("naive", "twap"):
            self.schedule: List[NanosecondTime] = sorted(schedule)
            n = len(self.schedule)
            self.slice_sizes: List[int] = build_slice_sizes(total_shares, n)
        else:
            self.schedule = []
            self.slice_sizes = []

        self.pov_frac = pov_frac
        self.pov_wake_freq_ns = pov_wake_freq_ns
        self.pov_lookback = pov_lookback
        self.start_time = start_time
        self.end_time = end_time

        self.fills: List[Tuple[NanosecondTime, int, int]] = []  # (time, price, qty)
        # one entry per place_market_order call: (time_placed, shares_requested) --
        # used downstream to bucket individual fills back into their originating slice
        self.slice_log: List[Tuple[NanosecondTime, int]] = []
        self.trading = False
        self.state = "AWAITING_WAKEUP"

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)
        self.oracle = self.kernel.oracle

    def get_wake_frequency(self) -> NanosecondTime:
        # Wake right at market open; wakeup() itself defers to the real schedule.
        return 0

    @property
    def remaining(self) -> int:
        return max(0, self.total_shares - self.total_filled)

    # ---- fill tracking -------------------------------------------------
    def order_executed(self, order) -> None:
        super().order_executed(order)
        self.fills.append((self.current_time, order.fill_price, order.quantity))
        self.total_filled += order.quantity

    # ---- naive / twap: no round-trip needed -----------------------------
    def _place_next_slice(self, current_time: NanosecondTime) -> None:
        # slice_sizes are precomputed (build_slice_sizes) to sum to exactly
        # total_shares and are disjoint per-slice allocations -- unlike
        # RLExecutionAgent's same-call chosen+forced pair, multiple slices placed
        # in this same while-loop (if several scheduled times are simultaneously
        # due) never double-request the same shares, so no within-call remaining
        # tracking is needed here; self.remaining only needs to reflect confirmed
        # fills for reporting/guard purposes.
        while self.schedule and current_time >= self.schedule[0]:
            self.schedule.pop(0)
            size = self.slice_sizes.pop(0)
            if size > 0 and self.remaining > 0:
                size = min(size, self.remaining)
                self.place_market_order(self.symbol, size, self.direction)
                self.slice_log.append((current_time, size))

        if self.schedule:
            self.set_wakeup(self.schedule[0])

    # ---- pov: needs the transacted-volume round trip --------------------
    def query_transacted_volume(self, symbol, bid_volume, ask_volume) -> None:
        super().query_transacted_volume(symbol, bid_volume, ask_volume)
        if self.state != "AWAITING_VOLUME" or self.remaining <= 0:
            return
        volume = bid_volume + ask_volume
        size = min(self.remaining, max(0, round(self.pov_frac * volume)))
        if size > 0:
            self.place_market_order(self.symbol, size, self.direction)
            self.slice_log.append((self.current_time, size))
        self.state = "AWAITING_WAKEUP"
        next_wake = self.current_time + self.pov_wake_freq_ns
        stop_time = self.end_time if self.end_time is not None else self.mkt_close
        if self.remaining > 0 and not self.mkt_closed and next_wake < stop_time:
            self.set_wakeup(next_wake)

    def wakeup(self, current_time: NanosecondTime) -> None:
        super().wakeup(current_time)

        if not self.mkt_open or not self.mkt_close:
            return
        self.trading = True

        if self.remaining <= 0 or self.mkt_closed:
            return

        if self.mode in ("naive", "twap"):
            self._place_next_slice(current_time)
        else:  # pov
            if self.start_time is not None and current_time < self.start_time:
                self.set_wakeup(self.start_time)
                return
            self.state = "AWAITING_VOLUME"
            self.get_transacted_volume(self.symbol, lookback_period=self.pov_lookback)
