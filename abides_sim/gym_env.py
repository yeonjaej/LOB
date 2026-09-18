"""A Gymnasium environment that trains an execution policy inside a *live* ABIDES
market, instead of replaying a static historical order book (as
`LOBdata_EDA.ipynb`'s Part 4/8/9 envs do).

The bridge between a synchronous Gym `step()`/`reset()` API and ABIDES's blocking,
single-call discrete-event `Kernel.run()` reuses a mechanism that already exists in
`abides_core.kernel.Kernel` itself (not the separate `abides-gym` package, which can't
be installed here -- its `__init__.py` unconditionally imports old `gym` and `ray`):

  - `Kernel.__init__` scans `agents` for any whose *direct* base classes include one
    named `"CoreGymAgent"` (checked by literal class name, not `isinstance`).
  - `Kernel.runner()` runs its normal event loop, but if dispatching a `WakeupMsg`
    causes that agent's `wakeup()` to return a non-`None` value, `runner()` does a
    bare early `return {"done": False, "result": wakeup_result}` -- the event queue
    and clock live on `self`, so calling `runner()` again just resumes the loop.
  - `runner(agent_actions=(gym_agent, action))` calls `gym_agent.apply_actions(action)`
    *before* resuming the loop -- this is how a chosen action gets injected at the
    exact simulated instant the previous `wakeup()` paused.

So `RLExecutionAgent.wakeup()` builds an observation and returns it (pausing the
kernel) instead of deciding its own trade; `apply_actions()` places the market order
for whatever action the external policy chose, immediately, before the background
market is allowed to advance any further.
"""

from collections import deque
from typing import Callable, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from abides_core.kernel import Kernel
from abides_core.utils import str_to_ns
from abides_markets.agents.trading_agent import TradingAgent
from abides_markets.messages.marketdata import L2SubReqMsg, L2DataMsg
from abides_markets.orders import Side

from .config import build_market_config

DEFAULT_ACTION_SHARES = {0: 0, 1: 500, 2: 2500, 3: 5000}

# v2: a real gap-in-the-action-space problem, confirmed two ways before fixing it --
# (1) both v1 checkpoints' actual action distribution on held-out seeds used ONLY
# actions 0 (hold) and 1 (500 shares), 0% usage of 2500/5000, and (2) measuring actual
# L1 ask-side resting depth (get_known_bid_ask) across live episodes found a median of
# just 48 shares, p90=166, p99=276 -- DEFAULT_ACTION_SHARES's smallest nonzero action
# (500) already exceeds the 99th percentile of what's typically resting, so the agent
# effectively never had a way to size a trade to the book it was actually facing. Bins
# below 350 densely bracket the empirical depth distribution (48th/76th/92nd/100th
# percentile at 50/100/200/350); above that, walking multiple levels is unavoidable at
# this order size regardless of bin choice, so the remaining bins climb geometrically
# up to the unchanged 5,000 ceiling rather than trying to densely cover a range where
# finer resolution wouldn't change which levels get walked anyway.
FINE_ACTION_SHARES = {
    0: 0, 1: 50, 2: 100, 3: 200, 4: 350, 5: 500,
    6: 750, 7: 1000, 8: 1500, 9: 2500, 10: 4000, 11: 5000,
}

# v3: even FINE_ACTION_SHARES only partially fixed the collapse (still concentrates
# on 2-3 of 12 bins; see LOBdata_RL_v2.ipynb section 5), and a 5-point ent_coef sweep
# (section 8/v2.1) found no exploration-pressure fix that beat the ent_coef=0
# baseline on the K=30 test benchmark either. Continuous removes the granularity
# ceiling entirely rather than choosing a denser-but-still-finite grid.
#
# v3.1: the first continuous attempt (0.5, i.e. a 5,000-share ceiling -- 20x the
# 250-share TWAP slice at horizon_steps=40) let early random exploration reach a
# region with genuinely catastrophic per-share economics, confirmed via an isolated
# single-trade slippage/fill-rate measurement: slip rises from 11.1bps/83% filled at
# the 250-share TWAP slice to 18.4bps/56% filled at 500 (2x), 22.3bps/44% at 750
# (3x), up to 28.9bps/24% filled at the old 5,000 ceiling -- a policy sampling near
# the [-1,1] boundary during early training gets hit with both a bad price and a
# mostly-unfilled order in the same step (see LOBdata_RL_v3.ipynb's front_loaded
# counterfactual, reward -17.72, for the training-time consequence). A full
# discounted-reward check ruled out gamma=0.99 procrastination-bias as the driver
# first (the gap barely closes under discounting) before landing on this as the
# better-supported mechanism. 0.05 (500 shares, 2x the TWAP slice) trades headroom
# for adaptive pacing against staying in the cheap/well-filled region of the curve.
#
# v3.3: this is now the DEFAULT for an explicit `max_action_frac` constructor
# parameter rather than a hardcoded global, because v3.2's trained policy sits at the
# 500-share ceiling on 56.9% of its decisions -- enough boundary saturation to ask
# whether the cap itself is binding, which needs a controlled retrain at a higher cap
# to answer. Changing this constant in place would silently re-evaluate every
# already-trained v3.1/v3.2 checkpoint under a ceiling it was never trained with --
# the same silent-mismatch bug class that already hit phi/psi and action_shares (see
# benchmark.py's run_ppo docstring). Keeping the default at 0.05 means every existing
# checkpoint keeps evaluating correctly with no call-site changes; a higher-cap run
# passes its own value explicitly.
MAX_ACTION_FRAC = 0.05

SESSION_START_S, SESSION_END_S = 9.5 * 3600, 16.0 * 3600  # 09:30:00-16:00:00

# Oracle input feature order -- shared verbatim between RLExecutionAgent's live
# per-wakeup computation (_market_state below) and oracle_train.py's offline
# per-episode feature generation, so a trained OraclePredictor can be called directly
# with no reordering step (a prior version manually reordered a 3-feature vector here;
# eliminated since it was a latent bug risk for no real benefit).
ORACLE_FEATURE_COLS = [
    "obi_l1", "obi_l5", "relative_depth_l5", "spread_bps", "micro_price_deviation_bps",
    "momentum_10s_bps", "momentum_60s_bps", "momentum_300s_bps",
    "delta_obi_10s", "realized_vol_60s_bps",
]

# v1: the policy's OWN base observation, independent of whether the oracle variant is
# active -- previously the policy only ever saw [spread_bps, obi_l1] of book state plus
# a single 1-step return; every one of these (minus momentum_300s_bps, deliberately --
# see LOBdata_RL_v1.ipynb) was already being computed for the oracle's benefit and
# simply never reached the policy. A strict subset of ORACLE_FEATURE_COLS by
# construction (see _market_state), so the two feature lists can never disagree about
# how a shared quantity (e.g. obi_l1) is computed.
BASE_FEATURE_COLS = [
    "spread_bps", "obi_l1", "obi_l5", "relative_depth_l5", "micro_price_deviation_bps",
    "momentum_10s_bps", "momentum_60s_bps", "delta_obi_10s", "realized_vol_60s_bps",
]


# v3.4: three features closing measured gaps between what the reward/action actually
# depend on and what the policy can see. Motivated by a direct variance decomposition
# of r_slip into an observable part (fill vs. contemporaneous mid -- driven by spread
# and depth, all already in BASE_FEATURE_COLS) and an unobservable part (how far the
# EWMA reference has fallen behind current price):
#
#   v3.1 (ref = prev mid): total per-step reward var 0.0007 -- the two parts cancel
#     almost exactly (cov -0.0219 vs component vars ~0.022), so the reward was nearly
#     deterministic. That, not learning quality, is why its explained_variance hit
#     0.83-0.94: the critic was fitting a near-constant target.
#   v3.2 (ref = EWMA a=0.13): total var 0.1880 (268x larger), DOMINATED by the
#     unobservable lag term (var 0.2009). The EWMA fix made the reward honest but
#     injected a large term absent from the observation -- capping the critic's fit
#     (explained_variance 0.2-0.35) and, worse, making the intended lesson ("I have
#     already pushed price, ease off") structurally unlearnable, since the agent has
#     no state variable through which to respond to the charge.
#
# Timescale gap this closes: the EWMA reference lags ~7.6 steps ~= 7.6 MINUTES, while
# the longest-horizon existing feature is momentum_60s_bps at 60 SECONDS. Nothing in
# the observation spanned minutes-to-episode scale, and nothing was self-referential
# (every market feature describes the book as it is, none says how far *this agent*
# moved it). l1_exec_side_depth additionally connects the action space to its own cost
# driver: the action is a fraction of total_shares, but slippage is governed by size
# RELATIVE TO resting depth, which the policy previously could not perceive.
EXTENDED_FEATURE_COLS = ["ref_lag_bps", "arrival_drift_bps", "log_l1_exec_side_depth"]


def base_obs_dim(extended_obs: bool = False) -> int:
    """time_left + inv_left (private state, not in BASE_FEATURE_COLS) + the market
    features, plus EXTENDED_FEATURE_COLS when `extended_obs`. Single source of truth
    for both RLExecutionAgent and ABIDESExecutionEnv's observation_space -- previously
    each independently hardcoded the same `5`, two places that could silently drift
    out of sync.

    `extended_obs` defaults False because changing the observation dimension is a
    HARD break for every existing checkpoint (the policy network's input layer shape
    is baked in at train time -- unlike phi/psi or max_action_frac, a mismatch here
    raises rather than silently mis-evaluating). Every v0-v3.3 checkpoint keeps
    loading against the 11-dim default; only runs that opt in get 14."""
    return 2 + len(BASE_FEATURE_COLS) + (len(EXTENDED_FEATURE_COLS) if extended_obs else 0)


def history_features(times: np.ndarray, mids: np.ndarray, obis: np.ndarray, t: int) -> dict:
    """Causal (no-lookahead) multi-horizon momentum/ΔOBI/realized-vol features derived
    from a tick history up to and including time `t`. `times` must be sorted ascending.

    Shared by `RLExecutionAgent` (live: `times`/`mids`/`obis` are an incrementally-built
    rolling buffer of actual L2 update ticks) and `oracle_train.py`'s offline feature
    generation (the full per-event series for a completed episode, queried at each
    decision-grid timestamp as if it were the live buffer at that instant) -- using one
    function for both guarantees identical feature definitions at train and serve time.

    Realized vol is the std of tick-to-tick returns among ticks actually observed in the
    trailing 60s window, not returns resampled onto an artificial 1-second grid: ticks
    arrive far faster than 1Hz in this simulation (confirmed: L2 updates fire on
    essentially every order-book-mutating event), so resampling to 1s would just discard
    information without changing what the estimator measures.
    """
    idx = int(np.searchsorted(times, t, side="right")) - 1
    if idx < 0:
        return {
            "momentum_10s_bps": 0.0, "momentum_60s_bps": 0.0, "momentum_300s_bps": 0.0,
            "delta_obi_10s": 0.0, "realized_vol_60s_bps": 0.0,
        }
    now_mid, now_obi = mids[idx], obis[idx]

    def _lookup(horizon_ns):
        j = int(np.searchsorted(times, t - horizon_ns, side="right")) - 1
        return j if j >= 0 else None

    def _momentum(horizon_ns):
        j = _lookup(horizon_ns)
        if j is None or not mids[j]:
            return 0.0
        return (now_mid - mids[j]) / mids[j] * 10_000

    j10 = _lookup(10_000_000_000)
    delta_obi_10s = (now_obi - obis[j10]) if j10 is not None else 0.0

    lo = int(np.searchsorted(times, t - 60_000_000_000, side="left"))
    window_mids = mids[lo:idx + 1]
    if len(window_mids) > 2:
        rets = np.diff(window_mids) / window_mids[:-1]
        realized_vol_60s_bps = float(np.std(rets) * 10_000)
    else:
        realized_vol_60s_bps = 0.0

    return {
        "momentum_10s_bps": _momentum(10_000_000_000),
        "momentum_60s_bps": _momentum(60_000_000_000),
        "momentum_300s_bps": _momentum(300_000_000_000),
        "delta_obi_10s": delta_obi_10s,
        "realized_vol_60s_bps": realized_vol_60s_bps,
    }


def secs_to_hms(total_s: float) -> str:
    total_s = int(total_s)
    return f"{total_s // 3600:02d}:{(total_s % 3600) // 60:02d}:{total_s % 60:02d}"


def pick_window(seed: int, window_minutes: int) -> "tuple[str, str]":
    """Same (start_time, end_time) string pair for a given (seed, window_minutes)
    regardless of caller -- shared by ABIDESExecutionEnv and the K-seed benchmark
    harness so every candidate strategy for a seed trades over the identical window,
    not just the identical background-market RNG stream."""
    rng = np.random.RandomState(seed)
    window_s = window_minutes * 60
    start_s = rng.uniform(SESSION_START_S, SESSION_END_S - window_s)
    return secs_to_hms(start_s), secs_to_hms(start_s + window_s)


class CoreGymAgent:
    """Marker mixin `Kernel` detects by literal base-class name (see module
    docstring) to enable the wakeup-return pause/resume mechanism. Deliberately not
    imported from `abides_gym`, which would drag in old `gym`/`ray`."""

    def update_raw_state(self) -> None:
        pass

    def get_raw_state(self):
        raise NotImplementedError


class RLExecutionAgent(TradingAgent, CoreGymAgent):
    """Buy-side execution agent controlled by an external policy via `apply_actions`.

    One "step" = one wakeup-pause-resume cycle. `total_shares` must be bought within
    `horizon_steps` decisions, spaced `step_interval_ns` apart; whatever remains after
    the last allowed decision is force-liquidated (with a reward penalty) rather than
    left unfilled, mirroring Part 4/8's forced-terminal-sweep design.
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
        total_shares: int = 10_000,
        horizon_steps: int = 20,
        step_interval_ns: int = 0,
        action_shares: Optional[dict] = None,
        continuous_action: bool = False,
        max_action_frac: float = MAX_ACTION_FRAC,
        extended_obs: bool = False,
        oracle_predict_fn: Optional[Callable[[np.ndarray], float]] = None,
        phi: float = 0.1,
        psi: float = 1.0,
        ewma_alpha: float = 0.13,
        direction: Side = Side.BID,
    ) -> None:
        """`phi`/`psi`: coefficients for the running inventory-urgency penalty and the
        terminal mandate-violation penalty (see `wakeup()`'s reward computation) --
        an Almgren-Chriss-style continuous-urgency reward, replacing the earlier flat
        forced-sweep multiplier so "wait, then dump everything at the end" is
        penalized throughout the episode, not just discovered at the last step.

        `ewma_alpha` (v3.2, attempt 3): `r_slip`/`r_term`'s reference price is an
        exponential moving average of the observed mid, updated every step, instead of
        just the previous step's mid. Two other designs were tried and rejected first:
        (1) the original "mid one step ago" reference resets every step, so it's blind
        to the agent's OWN cumulative price impact (confirmed: ~5.2bps/episode of real
        cost hidden from the v3.1 reward); (2) a terminal-step reconciliation against
        the true fixed-arrival cost was mathematically exact (verified: diff=0.000000
        against an independent recomputation) but broke training (`explained_variance`
        stuck near zero, `std` never narrowed over 150/1000 episodes) because
        `arrival_mid` mixes the agent's own impact with the market's own unrelated
        drift, and concentrating that mixed noise into one terminal spike turned out
        harder to learn from than the original problem; (3) referencing a same-seed,
        no-execution-agent "twin" simulation's mid path (uncontaminated by this
        agent's own fills, in principle) was implemented and sanity-checked before any
        training: the twin's own path is smooth, but it diverges from this agent's own
        kernel by tens of bps within 15-20 steps, and the divergence is
        INCONSISTENTLY SIGNED across seeds (a buy-side agent's real impact should push
        price one direction, not sometimes down) -- discrete-event-simulation chaos
        (the agent's first order perturbs event/match ordering, cascading into
        genuinely different background-agent outcomes between the two kernels), not
        attributable market impact, contaminating every step rather than just the
        terminal one. A pre-generated synthetic fundamental path (an independently-
        seeded OU-process draw) was considered next but not implemented: `config.py`'s
        `kappa=1.67e-16` makes that process behave as a plain random walk at this
        episode's timescale (~30-50bps of one-sigma movement per 45min window, per the
        module's own docstring), so an independent draw of it is a different
        realization than what actually happened that episode -- reintroducing noise of
        comparable size to what every attempt has been trying to eliminate, just
        relabeled. EWMA doesn't claim to eliminate the self-forgiveness problem the way
        attempts 2/3 aimed to -- it only dilutes it -- but for exactly that reason it
        can't fail the same way: it stays smooth and dense, with no new simulation and
        no risk of concentrating noise into a single spike.

        `continuous_action` (v3): switches `apply_actions` from a discrete
        `action_shares` dict lookup to interpreting `action` as a symmetric
        `[-1, 1]` Gym-facing value, affinely rescaled to a `[0, max_action_frac]`
        fraction of `total_shares` -- see `apply_actions` below. Additive; discrete
        remains the default so every v0/v1/v2 checkpoint's behavior is unchanged.

        `max_action_frac` (v3.3): the per-step cap, as a fraction of `total_shares`.
        Defaults to `MAX_ACTION_FRAC` (0.05 = 500 shares = 2x the TWAP slice), which
        every v3.1/v3.2 checkpoint was trained under -- must match training, exactly
        like phi/psi, or the policy is silently evaluated against a different action
        space than it learned.

        The Gym action space is `[-1, 1]`, not `[0, max_action_frac]` directly, even
        though the latter looks more natural -- SB3's Gaussian policy initializes
        with std~1, matched to a roughly-unit-scale, zero-centered range. Using the
        narrow, offset `[0, 0.5]` range as the Gym bounds directly made an initial
        std of ~0.8 (empirically observed) nearly twice the entire valid range, so
        almost every sampled action landed outside it and got clipped to one
        boundary -- confirmed directly: the first attempt's trained policy chose
        exactly 0 on 96% of decisions and >=0.49 (the max) on 2.6%, essentially a
        binary hold-or-dump-everything policy, worse than the discrete collapse it
        was meant to fix. Rescaling inside `apply_actions` instead keeps the Gym
        action space matched to the policy's natural output scale.
        """
        super().__init__(id, name, type, random_state, starting_cash, log_orders)
        self.symbol = symbol
        self.direction = direction
        self.total_shares = total_shares
        self.continuous_action = continuous_action
        self.max_action_frac = max_action_frac
        self.extended_obs = extended_obs
        # `remaining` is a derived property (see below), not a mutable field -- it
        # tracks CONFIRMED fills (self.total_filled, updated in order_executed), not
        # requested-but-possibly-unfilled order size. A market order can partially
        # fill if the opposite side of the book has less resting depth than
        # requested; decrementing "remaining" at request time (the previous design)
        # silently treated that shortfall as done, so remaining reaching 0 didn't
        # actually mean total_shares had been executed -- confirmed via a K=30
        # benchmark run where PPO under-filled on 11/30 seeds despite `remaining`
        # reaching exactly 0 in each case.
        self.total_filled = 0
        self.horizon_steps = horizon_steps
        self.step_interval_ns = step_interval_ns
        self.action_shares = action_shares or DEFAULT_ACTION_SHARES
        self.oracle_predict_fn = oracle_predict_fn
        self.phi = phi
        self.psi = psi
        self.ewma_alpha = ewma_alpha
        self._ewma_ref: Optional[float] = None

        self.n_obs = base_obs_dim(extended_obs) + (1 if oracle_predict_fn is not None else 0)
        self.step_count = 0
        self.fills = []  # (time, price, qty) -- full episode history, for diagnostics
        self.arrival_mid: Optional[float] = None  # kept for IS_bps diagnostics only
        self.prev_mid: Optional[float] = None
        self._subscribed = False
        # Rolling (time, mid, obi_l1) tick buffer for the oracle's multi-horizon
        # momentum/ΔOBI/realized-vol features (see `history_features`), populated on
        # every L2 update (handle_market_data override below) independent of this
        # agent's own wakeup cadence -- confirmed L2DataMsg pushes fire on essentially
        # every book-changing event, so this buffer has genuine sub-second resolution,
        # not just one entry per decision step. 6min lookback covers the 5min macro
        # momentum horizon plus margin.
        self._hist: deque = deque()
        self._hist_lookback_ns = str_to_ns("6min")
        # chosen-action fills (this step) vs. forced-terminal-sweep fills, tracked
        # separately so R_slip and R_term can each reference only their own shares
        self._step_cost, self._step_filled = 0.0, 0
        self._forced_cost, self._forced_filled = 0.0, 0
        self._forced = False

    def kernel_starting(self, start_time) -> None:
        super().kernel_starting(start_time)
        self.oracle = self.kernel.oracle
        # get_known_bid_ask() indexes these dicts directly (KeyError, not None, if
        # the symbol was never seen) -- pre-populate so the very first wakeup, before
        # any market-data message has arrived, doesn't crash.
        self.known_bids.setdefault(self.symbol, [])
        self.known_asks.setdefault(self.symbol, [])

    def get_wake_frequency(self):
        return 0

    @property
    def remaining(self) -> int:
        return max(0, self.total_shares - self.total_filled)

    def handle_market_data(self, message) -> None:
        super().handle_market_data(message)
        if not isinstance(message, L2DataMsg) or message.symbol != self.symbol:
            return
        bid, bid_vol, ask, ask_vol = self.get_known_bid_ask(self.symbol)
        if bid is None or ask is None:
            return
        mid = (bid + ask) / 2.0
        denom = (bid_vol or 0) + (ask_vol or 0)
        obi_l1 = (bid_vol - ask_vol) / denom if denom else 0.0
        self._hist.append((self.current_time, mid, obi_l1))
        cutoff = self.current_time - self._hist_lookback_ns
        while self._hist and self._hist[0][0] < cutoff:
            self._hist.popleft()

    def order_executed(self, order) -> None:
        super().order_executed(order)
        self.fills.append((self.current_time, order.fill_price, order.quantity))
        self.total_filled += order.quantity
        if order.tag == "forced":
            self._forced_cost += order.fill_price * order.quantity
            self._forced_filled += order.quantity
        else:
            self._step_cost += order.fill_price * order.quantity
            self._step_filled += order.quantity

    # ---- action injection: called by Kernel.runner((self, action)) BEFORE the
    # event loop resumes, i.e. at the exact simulated instant the previous wakeup()
    # paused. ----
    def apply_actions(self, action) -> None:
        is_last_step = (self.step_count + 1) >= self.horizon_steps
        self._forced = False
        true_remaining = self.remaining
        if self.continuous_action:
            # action[0] in [-1, 1] (the Gym-facing range) -> frac in [0, max_action_frac]
            unit = float(np.clip(action[0], -1.0, 1.0))
            frac = (unit + 1.0) / 2.0 * self.max_action_frac
            size = min(int(frac * self.total_shares), true_remaining)
        else:
            size = min(self.action_shares.get(int(action), 0), true_remaining)
        if size > 0:
            self.place_market_order(self.symbol, size, self.direction)
        # The chosen action and the forced clean-up are placed as two separate
        # orders (tagged "forced" on the second) so their fills can be attributed
        # to R_slip vs. R_term independently, matching the reward formulation. Both
        # can fire in this same call, before either's fill is confirmed (order_executed
        # hasn't run yet for the chosen-action order placed just above), so the
        # forced order's size is computed from true_remaining - size directly rather
        # than re-reading self.remaining -- which can't reflect the chosen order's
        # outcome yet -- to avoid requesting the same shares twice.
        if is_last_step:
            still_remaining = true_remaining - size
            if still_remaining > 0:
                self.place_market_order(self.symbol, still_remaining, self.direction, tag="forced")
                self._forced = True
        self.step_count += 1

    def _mid(self):
        bid, _, ask, _ = self.get_known_bid_ask(self.symbol)
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def _market_state(self, mid: float) -> dict:
        """Computes every book/flow-derived feature used by either the policy's base
        observation (`BASE_FEATURE_COLS`) or the oracle's own input
        (`ORACLE_FEATURE_COLS`), once per wakeup, from a single depth=5 snapshot plus
        the rolling tick-history buffer -- both feature lists just pull their own
        named subset out of the dict this returns, so a quantity like obi_l1 can never
        be computed two different ways in two different places and drift apart."""
        bid, bid_vol, ask, ask_vol = self.get_known_bid_ask(self.symbol)
        spread_bps = ((ask - bid) / mid * 10_000) if (bid and ask and mid) else 0.0
        denom1 = (bid_vol or 0) + (ask_vol or 0)
        obi_l1 = (bid_vol - ask_vol) / denom1 if denom1 else 0.0
        micro_dev_bps = 0.0
        if denom1 and mid:
            micro_price = (ask_vol * bid + bid_vol * ask) / denom1
            micro_dev_bps = (micro_price - mid) / mid * 10_000

        bids5 = self.known_bids.get(self.symbol, [])
        asks5 = self.known_asks.get(self.symbol, [])
        bid5_vol = float(sum(q for _, q in bids5))
        ask5_vol = float(sum(q for _, q in asks5))
        denom5 = bid5_vol + ask5_vol
        obi_l5 = (bid5_vol - ask5_vol) / denom5 if denom5 else 0.0
        relative_depth_l5 = (ask5_vol / bid5_vol) if bid5_vol else 1.0

        times = np.array([h[0] for h in self._hist], dtype=np.int64)
        mids_arr = np.array([h[1] for h in self._hist], dtype=np.float64)
        obis_arr = np.array([h[2] for h in self._hist], dtype=np.float64)
        hf = history_features(times, mids_arr, obis_arr, self.current_time)

        return {
            "spread_bps": spread_bps,
            "obi_l1": obi_l1,
            "obi_l5": obi_l5,
            "relative_depth_l5": relative_depth_l5,
            "micro_price_deviation_bps": micro_dev_bps,
            # v3.4: raw L1 depth on the side this agent actually consumes (ask side for
            # a buyer). log1p-compressed at observation time -- see EXTENDED_FEATURE_COLS.
            "l1_exec_side_depth": float(ask_vol or 0) if self.direction == Side.BID else float(bid_vol or 0),
            **hf,
        }

    def _build_obs(self, mid: float) -> np.ndarray:
        state = self._market_state(mid)
        time_left = 1.0 - (self.step_count / self.horizon_steps)
        inv_left = self.remaining / self.total_shares

        obs = [time_left, inv_left] + [state[f] for f in BASE_FEATURE_COLS]
        if self.extended_obs:
            # Reference lag: uses the PRE-update EWMA (self._ewma_ref is advanced later
            # in wakeup()), i.e. exactly the term that drove this step's r_slip. The
            # post-update lag would be (1-alpha) times this -- a constant rescaling, so
            # perfectly correlated and equally learnable; no reordering needed.
            ref = self._ewma_ref if self._ewma_ref is not None else mid
            ref_lag_bps = (mid - ref) / ref * 10_000 if ref else 0.0
            arrival = self.arrival_mid or mid
            arrival_drift_bps = (mid - arrival) / arrival * 10_000 if arrival else 0.0
            # log1p, not raw: measured L1 depth is heavy-tailed (median 48, p99 276,
            # rare bursts ~50k shares) and VecNormalize runs with norm_obs=False, so an
            # unscaled feature with 1000x outliers would sit alongside bps-scale
            # features and destabilize the MLP.
            obs += [ref_lag_bps, arrival_drift_bps, float(np.log1p(state["l1_exec_side_depth"]))]
        if self.oracle_predict_fn is not None:
            oracle_feats = np.array([state[f] for f in ORACLE_FEATURE_COLS], dtype=np.float64)
            obs.append(float(self.oracle_predict_fn(oracle_feats)))
        return np.array(obs, dtype=np.float32)

    def wakeup(self, current_time) -> Optional[dict]:
        super().wakeup(current_time)
        if not self.mkt_open or not self.mkt_close:
            return None

        if not self._subscribed:
            # L1SubReqMsg triggers an L1DataMsg, which TradingAgent.handle_market_data()
            # doesn't actually handle (it only recognizes L2DataMsg) -- L2 is what
            # actually keeps known_bids/known_asks populated. depth=5 (rather than 1)
            # so the oracle's L5-cumulative-OBI/relative-depth features are computable
            # live, matching what oracle_train.py generates offline.
            self.request_data_subscription(L2SubReqMsg(self.symbol, freq=1, depth=5))
            self._subscribed = True

        mid = self._mid()
        if mid is None:
            # Book not yet populated (only possible on the very first wakeup) --
            # nothing to report a reward against yet, and nothing meaningful to
            # observe; just try again shortly.
            self.set_wakeup(current_time + max(1, self.step_interval_ns // 20))
            return None

        if self.arrival_mid is None:
            self.arrival_mid = mid

        # pretrade_mid (v3.2, attempt 3): an EWMA of the observed mid -- see this
        # class's docstring for why (in short: "mid one step ago" resets every step
        # and is blind to this agent's own cumulative impact; a fixed `arrival_mid`
        # reference and two other attempted fixes each reintroduced a different kind
        # of hard-to-learn-from noise). Uses the EWMA's value BEFORE this step's
        # update -- "history up to but not including this step's own trade," matching
        # the original design's semantics -- then updates with the newly-observed
        # post-trade `mid` for next step's reference.
        if self._ewma_ref is None:
            self._ewma_ref = mid  # first wakeup: seed with arrival mid, nothing to average yet
        pretrade_mid = self._ewma_ref

        # is_bps stays arrival-relative (diagnostic only, comparable to the rest of
        # this project's IS convention); reward_bps is the three-part training signal:
        # R_slip (execution cost vs. the EWMA reference mid, size-weighted) + R_inv
        # (continuous Almgren-Chriss inventory-urgency penalty, every step, so "wait
        # then dump at the end" has a running cost rather than only a terminal one) +
        # R_term (forced-sweep slippage plus an explicit mandate-violation penalty, on
        # the terminal step only).
        is_bps = 0.0
        if self._step_filled > 0:
            avg_price = self._step_cost / self._step_filled
            is_bps = (avg_price - self.arrival_mid) / self.arrival_mid * 10_000

        r_slip = 0.0
        if self._step_filled > 0:
            avg_price = self._step_cost / self._step_filled
            slip_bps = (avg_price - pretrade_mid) / pretrade_mid * 10_000
            r_slip = -slip_bps * (self._step_filled / self.total_shares)

        i_t = self.remaining / self.total_shares
        r_inv = -self.phi * (i_t ** 2)

        r_term = 0.0
        if self._forced and self._forced_filled > 0:
            forced_avg_price = self._forced_cost / self._forced_filled
            forced_slip_bps = (forced_avg_price - pretrade_mid) / pretrade_mid * 10_000
            i_T = self._forced_filled / self.total_shares
            r_term = -forced_slip_bps * i_T - self.psi * (i_T ** 2)

        reward_bps = r_slip + r_inv + r_term

        terminated = self.remaining <= 0 or self.step_count >= self.horizon_steps

        obs = self._build_obs(mid)
        self.prev_mid = mid  # kept only as a last-known-mid fallback for get_raw_state()
        self._ewma_ref = self.ewma_alpha * mid + (1 - self.ewma_alpha) * self._ewma_ref
        raw_state = {
            "obs": obs,
            "mid": mid,
            "arrival_mid": self.arrival_mid,
            "step_cost": self._step_cost + self._forced_cost,
            "step_filled": self._step_filled + self._forced_filled,
            "is_bps": is_bps,
            "reward_bps": reward_bps,
            "forced": self._forced,
            "terminated": bool(terminated),
        }
        self._step_cost, self._step_filled = 0.0, 0
        self._forced_cost, self._forced_filled = 0.0, 0
        self._forced = False

        if not terminated:
            self.set_wakeup(current_time + self.step_interval_ns)

        return raw_state

    def update_raw_state(self) -> None:
        pass

    def get_raw_state(self) -> dict:
        # Kernel.runner() calls this at final "done" (event queue empty / stop_time
        # reached) instead of via wakeup() -- can happen if the episode's kernel
        # stop_time arrives before horizon_steps is reached for any reason.
        mid = self._mid() or self.prev_mid or self.arrival_mid
        return {
            "obs": np.zeros(self.n_obs, dtype=np.float32),
            "mid": mid,
            "arrival_mid": self.arrival_mid,
            "step_cost": 0.0,
            "step_filled": 0,
            "is_bps": 0.0,
            "reward_bps": 0.0,
            "forced": False,
            "terminated": True,
        }


class ABIDESExecutionEnv(gym.Env):
    """One episode = one short-window ABIDES kernel run (background market only;
    the RL agent is the only "execution" activity). Reward per step is
    R_slip + R_inv + R_term (see `RLExecutionAgent.wakeup()`): execution cost vs. an
    EWMA-smoothed reference price, a continuous inventory-urgency penalty, and a
    terminal mandate-violation penalty for whatever had to be force-liquidated -- see
    `RLExecutionAgent.__init__`'s docstring for why EWMA, and what two other designs
    (terminal reconciliation against `arrival_mid`; a same-seed no-agent "twin"
    simulation) were tried and rejected first.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        total_shares: int = 10_000,
        horizon_steps: int = 20,
        window_minutes: int = 45,
        step_interval: str = "1min",
        action_shares: Optional[dict] = None,
        continuous_action: bool = False,
        max_action_frac: float = MAX_ACTION_FRAC,
        extended_obs: bool = False,
        oracle_predict_fn: Optional[Callable[[np.ndarray], float]] = None,
        phi: float = 0.1,
        psi: float = 1.0,
        market_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        from abides_core.utils import str_to_ns

        self.total_shares = total_shares
        self.horizon_steps = horizon_steps
        self.window_minutes = window_minutes
        self.step_interval_ns = str_to_ns(step_interval)
        self.action_shares = action_shares or DEFAULT_ACTION_SHARES
        self.continuous_action = continuous_action
        self.max_action_frac = max_action_frac
        self.extended_obs = extended_obs
        self.oracle_predict_fn = oracle_predict_fn
        self.phi = phi
        self.psi = psi
        self.market_kwargs = market_kwargs or {}

        n_obs = base_obs_dim(extended_obs) + (1 if oracle_predict_fn is not None else 0)
        if continuous_action:
            # [-1, 1], not [0, max_action_frac] -- see RLExecutionAgent.__init__'s
            # docstring for why the Gym-facing range is symmetric/unit-scale while
            # the actual rescaling to a share fraction happens in apply_actions.
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(len(self.action_shares))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

        self.kernel: Optional[Kernel] = None
        self.gym_agent: Optional[RLExecutionAgent] = None
        self.arrival_price: Optional[float] = None

    def _build_kernel(self, seed: int) -> Kernel:
        # Randomize the session's clock position each episode (fixed window length)
        # so the policy doesn't overfit to one time-of-day's dynamics.
        start_time, end_time = pick_window(seed, self.window_minutes)

        def build_exec_agent(agent_id):
            self.gym_agent = RLExecutionAgent(
                id=agent_id,
                name="RL_EXECUTION_AGENT",
                type="ExecutionAgent",
                symbol=self.market_kwargs.get("symbol", "ABM"),
                total_shares=self.total_shares,
                horizon_steps=self.horizon_steps,
                step_interval_ns=self.step_interval_ns,
                action_shares=self.action_shares,
                continuous_action=self.continuous_action,
                max_action_frac=self.max_action_frac,
                extended_obs=self.extended_obs,
                oracle_predict_fn=self.oracle_predict_fn,
                phi=self.phi,
                psi=self.psi,
                log_orders=True,
            )
            return self.gym_agent

        config = build_market_config(
            seed=seed,
            start_time=start_time,
            end_time=end_time,
            exec_agent_builder=build_exec_agent,
            **{k: v for k, v in self.market_kwargs.items() if k != "symbol"},
        )
        return Kernel(
            random_state=config["random_state_kernel"],
            agents=config["agents"],
            start_time=config["start_time"],
            stop_time=config["stop_time"],
            agent_latency_model=config["agent_latency_model"],
            default_computation_delay=config["default_computation_delay"],
            custom_properties=config["custom_properties"],
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        episode_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        self.kernel = self._build_kernel(episode_seed)
        self.kernel.initialize()
        result = self.kernel.runner()
        raw = result["result"]
        self.arrival_price = raw["arrival_mid"]
        return raw["obs"], {}

    def step(self, action):
        result = self.kernel.runner((self.gym_agent, action))
        raw = result["result"]
        # reward_bps is already reward-signed (higher = better) -- R_slip is positive
        # for favorable execution, R_inv/R_term are penalties baked in as <= 0 -- so,
        # unlike the old is_bps-derived reward (which was cost-positive and needed
        # negating here), this one must NOT be negated again. See
        # RLExecutionAgent.wakeup() for the full R_slip + R_inv + R_term formulation.
        reward = raw["reward_bps"]
        info = {
            "cost": raw["step_cost"],
            "filled": raw["step_filled"],
            "is_bps": raw["is_bps"],
            "forced": raw["forced"],
            "mid": raw["mid"],
        }
        terminated = bool(raw["terminated"]) or bool(result["done"])
        return raw["obs"], reward, terminated, False, info
