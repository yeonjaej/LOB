"""Train the "regression prediction" oracle (Part 9.1's ridge-regression concept in
`LOBdata_EDA.ipynb`) on ABIDES-generated data instead of real LOBSTER data -- a
LOBSTER-fitted regression has no reason to predict returns well in a structurally
different synthetic market.

Feature set matches `ORACLE_FEATURE_COLS` in `gym_env.py`: obi_l1, obi_l5,
relative_depth_l5, spread_bps, micro_price_deviation_bps, momentum at 10s/60s/300s,
ΔOBI over 10s, and 60s realized vol. `RLExecutionAgent` now subscribes at L2 depth=5
and maintains a rolling tick buffer specifically so it can compute this same feature
set live (see `gym_env.py`'s `_oracle_features`/`history_features`) -- this function
reconstructs the identical features offline from a completed episode's full per-event
book history, so train and serve see the same information.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from scipy.stats import skew as _skew

from abides_core import abides
from .config import build_market_config
from .gym_env import ORACLE_FEATURE_COLS, history_features


def generate_clean_episode_features(
    seed: int,
    symbol: str = "ABM",
    window_minutes: int = 45,
    sample_interval: str = "2min",
    **market_kwargs,
) -> pd.DataFrame:
    """Runs one background-market-only (no execution agent) episode and returns a
    DataFrame of `ORACLE_FEATURE_COLS` + `forward_return_bps`, one row per
    `sample_interval` decision-grid point (matching the live agent's step cadence),
    each row's features computed causally from data at-or-before that point only."""
    from abides_core.utils import str_to_ns

    rng = np.random.RandomState(seed)
    session_start_s, session_end_s = 9.5 * 3600, 16.0 * 3600
    window_s = window_minutes * 60
    start_s = rng.uniform(session_start_s, session_end_s - window_s)
    start_time = f"{int(start_s) // 3600:02d}:{(int(start_s) % 3600) // 60:02d}:{int(start_s) % 60:02d}"
    end_s = start_s + window_s
    end_time = f"{int(end_s) // 3600:02d}:{(int(end_s) % 3600) // 60:02d}:{int(end_s) % 60:02d}"

    config = build_market_config(
        seed=seed, symbol=symbol, start_time=start_time, end_time=end_time, **market_kwargs
    )
    end_state = abides.run(config)
    ob = end_state["agents"][0].order_books[symbol]
    empty = pd.DataFrame(columns=ORACLE_FEATURE_COLS + ["forward_return_bps"])

    l2 = ob.get_L2_snapshots(nlevels=5)
    times = l2["times"].astype("int64")
    if len(times) < 5:
        return empty
    order = np.argsort(times, kind="stable")
    times, bids, asks = times[order], l2["bids"][order], l2["asks"][order]

    bid1_price, bid1_size = bids[:, 0, 0].astype(float), bids[:, 0, 1].astype(float)
    ask1_price, ask1_size = asks[:, 0, 0].astype(float), asks[:, 0, 1].astype(float)

    # bids_padding/asks_padding (inside get_L2_snapshots) fill a completely EMPTY book
    # side with price=0, qty=0 rather than leaving it undefined -- confirmed empirically
    # ~2.6% of ticks in a typical episode (clustered at session open before two-sided
    # liquidity forms, and again near close as noise agents wind down). Treating that 0
    # as a real price silently produces a ~half-price "mid" and an exact +/-20000bps
    # "spread" at those ticks, which is enough badly-wrong rows to dominate correlation
    # statistics computed over a small per-episode sample (this is what originally
    # produced an apparent -1.0000 correlation between micro_price_deviation_bps and
    # forward_return_bps -- traced to exactly these degenerate rows, not a real
    # relationship). The live agent never hits this: TradingAgent.get_known_bid_ask()
    # returns None for an empty side (its L2 subscription filters to positive-quantity
    # levels only), and RLExecutionAgent's handle_market_data override already guards
    # on that None. Offline, drop these ticks entirely -- from history too, not just
    # the final training rows, since a degenerate tick sitting in the momentum/
    # realized-vol lookback window would corrupt those features as well.
    valid = (bid1_price > 0) & (ask1_price > 0)
    times, bids, asks = times[valid], bids[valid], asks[valid]
    bid1_price, bid1_size = bid1_price[valid], bid1_size[valid]
    ask1_price, ask1_size = ask1_price[valid], ask1_size[valid]
    if len(times) < 5:
        return empty

    mids = (bid1_price + ask1_price) / 2.0
    denom1 = bid1_size + ask1_size
    bid5_vol = bids[:, :, 1].sum(axis=1).astype(float)
    ask5_vol = asks[:, :, 1].sum(axis=1).astype(float)
    denom5 = bid5_vol + ask5_vol

    # np.where(cond, a/b, c) still eagerly evaluates a/b (and warns) for every index
    # even where cond is False -- np.errstate silences that expected div-by-zero/NaN
    # noise on the branches np.where is about to discard anyway.
    with np.errstate(divide="ignore", invalid="ignore"):
        obi_l1 = np.where(denom1 > 0, (bid1_size - ask1_size) / denom1, 0.0)
        spread_bps = np.where(mids > 0, (ask1_price - bid1_price) / mids * 10_000, 0.0)
        micro_price = np.where(denom1 > 0, (ask1_size * bid1_price + bid1_size * ask1_price) / denom1, mids)
        micro_dev_bps = np.where(mids > 0, (micro_price - mids) / mids * 10_000, 0.0)
        # bids_padding/asks_padding (called inside get_L2_snapshots) fill any missing
        # level with qty=0, so summing across all `nlevels` columns is exactly "total
        # volume across whatever levels actually exist, up to 5" -- the same quantity
        # the live agent computes by summing its (possibly-fewer-than-5)
        # known_bids/known_asks.
        obi_l5 = np.where(denom5 > 0, (bid5_vol - ask5_vol) / denom5, 0.0)
        relative_depth_l5 = np.where(bid5_vol > 0, ask5_vol / bid5_vol, 1.0)

    step_ns = str_to_ns(sample_interval)
    t0, t1 = int(times.min()), int(times.max())
    # Every grid point needs a genuine future observation within the recorded window
    # for its forward_return_bps label -- stop `step_ns` short of the last tick.
    grid = np.arange(t0, t1 - step_ns, step_ns)
    if len(grid) < 4:
        return empty

    def _mid_at(t):
        idx = int(np.searchsorted(times, t, side="right")) - 1
        return mids[max(idx, 0)]

    rows = []
    for t in grid:
        idx = int(np.searchsorted(times, t, side="right")) - 1
        if idx < 0:
            continue
        hf = history_features(times, mids, obi_l1, t)
        forward_mid = _mid_at(t + step_ns)
        now_mid = mids[idx]
        rows.append({
            "obi_l1": obi_l1[idx],
            "obi_l5": obi_l5[idx],
            "relative_depth_l5": relative_depth_l5[idx],
            "spread_bps": spread_bps[idx],
            "micro_price_deviation_bps": micro_dev_bps[idx],
            "momentum_10s_bps": hf["momentum_10s_bps"],
            "momentum_60s_bps": hf["momentum_60s_bps"],
            "momentum_300s_bps": hf["momentum_300s_bps"],
            "delta_obi_10s": hf["delta_obi_10s"],
            "realized_vol_60s_bps": hf["realized_vol_60s_bps"],
            "forward_return_bps": (forward_mid - now_mid) / now_mid * 10_000 if now_mid else np.nan,
        })

    if not rows:
        return empty
    return pd.DataFrame(rows)[ORACLE_FEATURE_COLS + ["forward_return_bps"]].dropna()


def train_oracle(seeds, n_workers: Optional[int] = None, **market_kwargs):
    """Returns (predict_fn, diagnostics: dict). predict_fn(features_array) -> bps.

    Episode generation dominates wall-clock (each episode is a full ABIDES run; ridge
    fitting itself is ~milliseconds regardless of how many alphas are tried) and the
    episodes are fully independent, so they're farmed out across a process pool rather
    than run in a loop.
    """
    t0 = time.time()
    n_workers = min(n_workers or max(1, (os.cpu_count() or 2) - 1), len(seeds))
    worker_fn = partial(generate_clean_episode_features, **market_kwargs)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(worker_fn, seeds))
    frames = []
    for i, df in enumerate(results):
        df = df.copy()
        df["episode"] = i
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    print(f"Generated {len(data)} training rows from {len(seeds)} episodes in {time.time()-t0:.1f}s ({n_workers} workers)")

    # Episode-grouped 70/30 train/test split: test episodes stay untouched until the
    # final test_corr below. Alpha selection must NOT reuse the training data it fit
    # on (that systematically favors the least-regularized alpha, since lower alpha
    # generically fits its own training data at least as well) -- so the train
    # episodes are further split into a fit/validation subset, grouped the same way,
    # purely for alpha selection.
    n_episodes = data["episode"].nunique()
    test_start = int(n_episodes * 0.7)
    train_episode_ids = list(range(test_start))
    test_mask = data["episode"].isin(range(test_start, n_episodes))

    val_start = max(1, int(len(train_episode_ids) * 0.7))
    fit_episode_ids = set(train_episode_ids[:val_start])
    val_episode_ids = set(train_episode_ids[val_start:])
    train_mask = data["episode"].isin(train_episode_ids)
    fit_mask = data["episode"].isin(fit_episode_ids)
    val_mask = data["episode"].isin(val_episode_ids)

    feature_cols = ORACLE_FEATURE_COLS
    X = data[feature_cols].to_numpy(dtype=np.float64)
    Y = data["forward_return_bps"].to_numpy(dtype=np.float64)
    X_train, Y_train = X[train_mask], Y[train_mask]
    X_test, Y_test = X[test_mask], Y[test_mask]
    X_fit, Y_fit = X[fit_mask], Y[fit_mask]
    X_val, Y_val = X[val_mask], Y[val_mask]

    fit_ct, fit_order, fit_choice = _build_normalizer(X_fit, feature_cols)

    best_alpha, best_val_score = None, -np.inf
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        m = Ridge(alpha=alpha)
        m.fit(fit_ct.transform(X_fit), Y_fit)
        val_score = (
            np.corrcoef(m.predict(fit_ct.transform(X_val)), Y_val)[0, 1]
            if len(Y_val) > 1 else -np.inf
        )
        if val_score > best_val_score:
            best_alpha, best_val_score = alpha, val_score

    # Refit on the full train split (fit+val episodes) with the chosen alpha -- more
    # data than the fit-only subset, and its own normalizer fit fresh on the full
    # train set rather than leaking any test-episode statistics. Skew is re-measured
    # here (not reused from the fit-only subset) since it can shift with more data.
    train_ct, train_order, train_choice = _build_normalizer(X_train, feature_cols)
    best_model = Ridge(alpha=best_alpha)
    best_model.fit(train_ct.transform(X_train), Y_train)
    train_corr = np.corrcoef(best_model.predict(train_ct.transform(X_train)), Y_train)[0, 1]

    test_pred = best_model.predict(train_ct.transform(X_test))
    test_corr = np.corrcoef(test_pred, Y_test)[0, 1] if len(Y_test) > 1 else float("nan")

    diagnostics = {
        "n_rows": len(data),
        "n_episodes": n_episodes,
        "alpha": best_alpha,
        "val_corr": best_val_score,
        "train_corr": train_corr,
        "test_corr": test_corr,
        "normalization": train_choice,
        "coefficients": dict(zip(train_order, best_model.coef_)),
    }

    return OraclePredictor(best_model, train_ct), diagnostics


def _build_normalizer(X_fit: np.ndarray, feature_cols: list):
    """Per-feature normalization chosen by measured |skew|, not one blanket transform:
    |skew|<=1.0 (symmetric to moderately skewed) gets z-score standardization; >1.0
    (heavy-tailed -- e.g. spread_bps's near-constant-plus-rare-spikes shape, skew~15)
    gets rank-based QuantileTransformer, whose outlier leverage is bounded by
    construction. Thresholds measured fresh on whatever data is actually being fit,
    not hardcoded from a prior feature set.

    Returns (fitted ColumnTransformer, output column order, {feature: "rank"|"zscore"}).
    ColumnTransformer concatenates its named transformers' outputs in the order they're
    listed (all "rank" columns, then all "zscore" columns) rather than the original
    column order -- `output order` tracks that so diagnostics can label coefficients
    correctly; the model itself doesn't care about column order as long as fit/predict
    both go through the same fitted `ct`.
    """
    skews = np.abs(_skew(X_fit, axis=0, nan_policy="omit"))
    rank_idx = [i for i, s in enumerate(skews) if s > 1.0]
    std_idx = [i for i, s in enumerate(skews) if s <= 1.0]
    transformers = []
    if rank_idx:
        transformers.append((
            "rank",
            QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(X_fit)), random_state=0),
            rank_idx,
        ))
    if std_idx:
        transformers.append(("zscore", StandardScaler(), std_idx))
    ct = ColumnTransformer(transformers).fit(X_fit)
    output_order = [feature_cols[i] for i in rank_idx] + [feature_cols[i] for i in std_idx]
    choice = {feature_cols[i]: "rank" for i in rank_idx}
    choice.update({feature_cols[i]: "zscore" for i in std_idx})
    return ct, output_order, choice


class OraclePredictor:
    """A plain, module-level, picklable callable -- unlike a closure (train_oracle's
    previous return type), this survives `pickle.dump`/`.load()` across processes, so
    a fitted oracle doesn't need to be retrained just to run in a different script.
    """

    def __init__(self, model: Ridge, ct: ColumnTransformer):
        self.model = model
        self.ct = ct

    def __call__(self, features: np.ndarray) -> float:
        # features: RLExecutionAgent._oracle_features's 10-vector, already built in
        # ORACLE_FEATURE_COLS order -- no reordering needed (a prior 3-feature version
        # manually reordered here, which was itself a latent bug risk).
        x = np.asarray(features, dtype=np.float64).reshape(1, -1)
        return float(self.model.predict(self.ct.transform(x))[0])
