import hydra
from omegaconf import DictConfig

import numpy as np
from rich import print as printr

import gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


@hydra.main("configs", "base", version_base="1.1")
def main(cfg: DictConfig) -> float:
    printr(cfg)

    env = gym.make(cfg.env_id)
    model = SAC("MlpPolicy", env, verbose=cfg.verbose, tensorboard_log=cfg.log_dir, seed=cfg.seed)
    model.learn(total_timesteps=cfg.total_timesteps)
    model.save(cfg.model_fn)

    # Evaluate
    env = Monitor(gym.make(cfg.env_id))
    means, stds = evaluate_policy(model, env, n_eval_episodes=cfg.n_eval_episodes)
    performance = np.mean(means)

    return performance


if __name__ == "__main__":
    main()
