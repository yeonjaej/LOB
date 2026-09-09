"""Train a PPO execution policy against `ABIDESExecutionEnv`, with progress logging
and periodic checkpointing so a long background run can be monitored and survives a
crash without losing everything.
"""

import time
from functools import partial
from typing import Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from .gym_env import ABIDESExecutionEnv


def _make_single_env(env_kwargs: dict):
    """Module-level (not a nested closure) so it's picklable -- SubprocVecEnv's spawned
    worker processes re-import this module and look the function up by qualified name;
    a closure defined inside train_ppo_agent can't be pickled that way."""
    return Monitor(ABIDESExecutionEnv(**env_kwargs))


class EpisodeProgressCallback(BaseCallback):
    """Stops training once `target_episodes` episodes have completed (rather than a
    fixed timestep count, which is hard to predict when episode length varies with
    the policy's own behavior). Prints a progress line and checkpoints every
    `print_every` episodes.

    With n_envs>1, `self.locals["infos"]` carries one entry per parallel env at each
    call, so multiple episodes can complete in the same `_on_step` -- the loop below
    already sums across all of them correctly, but can overshoot `target_episodes` by
    up to (n_envs-1) if several envs finish in the same joint step. Negligible against
    a several-hundred-episode budget; not worth the extra bookkeeping to trim exactly.
    """

    def __init__(
        self, target_episodes: int, checkpoint_path: str, print_every: int = 25,
        snapshot_every: Optional[int] = None, verbose: int = 0,
    ):
        super().__init__(verbose)
        self.target_episodes = target_episodes
        self.checkpoint_path = checkpoint_path
        self.print_every = print_every
        # snapshot_every: in addition to the rolling crash-safety overwrite of
        # checkpoint_path above, also save a permanently-numbered copy every N
        # episodes (e.g. "{checkpoint_path}_ep300") so a longer run can be evaluated
        # at multiple points along training and the best one picked afterward,
        # instead of just trusting the final (possibly overfit/regressed) snapshot.
        self.snapshot_every = snapshot_every
        self.episode_count = 0
        self.recent_rewards = []
        self.t0 = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.episode_count += 1
                self.recent_rewards.append(ep["r"])
                if self.episode_count % self.print_every == 0:
                    recent = self.recent_rewards[-self.print_every:]
                    print(
                        f"[episode {self.episode_count}/{self.target_episodes}] "
                        f"mean_reward(last {len(recent)})={np.mean(recent):.3f} "
                        f"elapsed={time.time() - self.t0:.1f}s",
                        flush=True,
                    )
                    self.model.save(self.checkpoint_path)
                if self.snapshot_every and self.episode_count % self.snapshot_every == 0:
                    self.model.save(f"{self.checkpoint_path}_ep{self.episode_count}")
                if self.episode_count >= self.target_episodes:
                    return False
        return True


def train_ppo_agent(
    target_episodes: int,
    seed: int = 0,
    checkpoint_path: str = "ppo_checkpoint",
    env_kwargs: Optional[dict] = None,
    tensorboard_log: Optional[str] = "tb_logs",
    run_name: Optional[str] = None,
    n_envs: int = 1,
    snapshot_every: Optional[int] = None,
) -> PPO:
    # oracle_predict_fn (if used) goes through env_kwargs, not a separate top-level
    # parameter -- a prior version had both, which crashed with a duplicate-keyword
    # TypeError since ABIDESExecutionEnv(oracle_predict_fn=..., **env_kwargs) collides
    # whenever env_kwargs also carries it (or even just via this function's own
    # oracle_predict_fn=None default). One way to specify it, not two.
    #
    # Monitor wraps the raw env so EpisodeProgressCallback still sees genuine bps-scale
    # episode rewards for logging; VecNormalize sits outside it and only rescales what
    # PPO's rollout buffer sees for the advantage/value calculation -- it never touches
    # observations (norm_obs=False), so it needs no saving/restoring for later inference
    # or evaluation, only for training dynamics.
    env_kwargs = env_kwargs or {}
    env_fns = [partial(_make_single_env, env_kwargs) for _ in range(n_envs)]
    # n_envs>1 runs each ABIDES episode in its own OS process (SubprocVecEnv) -- each
    # simulation is CPU-bound Python (discrete-event loop, not I/O), so genuine
    # multi-process parallelism is what actually cuts wall-clock, not threads.
    # start_method="spawn" explicitly rather than SB3's platform-dependent default,
    # since fork-based multiprocessing after certain library initialization is known to
    # be unsafe on macOS.
    base_env = SubprocVecEnv(env_fns, start_method="spawn") if n_envs > 1 else DummyVecEnv(env_fns)
    vec_env = VecNormalize(base_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # n_steps is PER ENV in SB3 (total rollout buffer = n_steps * n_envs) -- scaled down
    # as n_envs grows so the buffer size, and therefore the update cadence in
    # episodes-per-gradient-update, stays close to the single-env tuning already
    # validated (n_steps=256, n_envs=1 -- the 500-episode no-oracle run that showed
    # explained_variance moving into 0.6-0.8) rather than silently becoming n_envs times
    # coarser. Floor of 32 keeps enough per-env diversity in one rollout batch.
    n_steps = max(256 // n_envs, 32)
    batch_size = 64 if (n_steps * n_envs) % 64 == 0 else n_steps * n_envs
    model = PPO(
        "MlpPolicy", vec_env, n_steps=n_steps, batch_size=batch_size, verbose=1, seed=seed,
        tensorboard_log=tensorboard_log,
    )
    callback = EpisodeProgressCallback(
        target_episodes=target_episodes, checkpoint_path=checkpoint_path, snapshot_every=snapshot_every,
    )

    t0 = time.time()
    # Generous upper bound on timesteps; the callback stops training on episode count.
    # tb_log_name gives each run its own named subdirectory under tensorboard_log, so
    # multiple runs (e.g. no-oracle vs. with-oracle) overlay as separate curves in the
    # same TensorBoard instance instead of overwriting each other.
    model.learn(
        total_timesteps=target_episodes * 100, callback=callback,
        tb_log_name=run_name or checkpoint_path,
    )
    print(f"Finished: {callback.episode_count} episodes in {time.time() - t0:.1f}s ({n_envs} envs)")

    model.save(checkpoint_path)
    return model
