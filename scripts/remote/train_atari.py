#!/usr/bin/env python3
"""Single-file Atari training script for autoresearcher2 experiments.

Trains PPO on an Atari game using stable-baselines3, prints final stats
in a parseable format matching the train.py output convention.

Usage:
    python train_atari.py --game Breakout --lr 5e-4 --network-size medium
    python train_atari.py --game Pong --lr 1e-3 --network-size large --total-timesteps 500000

Prerequisites:
    pip install gymnasium[atari] ale-py stable-baselines3
"""

import argparse
import time

import ale_py
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# Register ALE Atari environments
gym.register_envs(ale_py)
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# ---------------------------------------------------------------------------
# Network architectures
# ---------------------------------------------------------------------------

NETWORK_CONFIGS = {
    "small": dict(pi=[64, 64], vf=[64, 64]),
    "medium": dict(pi=[256, 256], vf=[256, 256]),
    "large": dict(pi=[512, 256, 128], vf=[512, 256, 128]),
}

GAME_TO_ENV = {
    "Breakout": "BreakoutNoFrameskip-v4",
    "SpaceInvaders": "SpaceInvadersNoFrameskip-v4",
    "Pong": "PongNoFrameskip-v4",
}


def make_env(game: str):
    """Create a wrapped Atari environment."""
    env_id = GAME_TO_ENV.get(game, f"{game}NoFrameskip-v4")

    def _init():
        env = gym.make(env_id)
        env = AtariWrapper(env)
        return env

    return _init


def evaluate(model, game: str, n_episodes: int = 20) -> float:
    """Evaluate the trained model over n episodes, return mean reward."""
    env = DummyVecEnv([make_env(game)])
    env = VecFrameStack(env, n_stack=4)

    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            if done[0]:
                break
        rewards.append(total_reward)
    env.close()

    return float(np.mean(rewards))


def main():
    parser = argparse.ArgumentParser(description="Train PPO on Atari")
    parser.add_argument("--game", type=str, default="Breakout",
                        choices=list(GAME_TO_ENV.keys()))
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--network-size", type=str, default="medium",
                        choices=list(NETWORK_CONFIGS.keys()))
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=20)
    args = parser.parse_args()

    net_arch = NETWORK_CONFIGS[args.network_size]

    print(f"Game: {args.game}")
    print(f"Learning rate: {args.lr}")
    print(f"Network size: {args.network_size} ({net_arch})")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Num envs: {args.n_envs}")

    # Create vectorized environment
    env = DummyVecEnv([make_env(args.game) for _ in range(args.n_envs)])
    env = VecFrameStack(env, n_stack=4)

    # Create PPO agent
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=args.lr,
        n_steps=128,
        n_epochs=4,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=net_arch),
        verbose=0,
        seed=42,
    )

    # Train
    t_start = time.time()
    model.learn(total_timesteps=args.total_timesteps)
    training_time = time.time() - t_start

    env.close()

    # Evaluate
    mean_reward = evaluate(model, args.game, n_episodes=args.eval_episodes)

    # Print results in parseable format
    fps = args.total_timesteps / training_time if training_time > 0 else 0
    print("---")
    print(f"mean_reward:     {mean_reward:.2f}")
    print(f"total_timesteps: {args.total_timesteps}")
    print(f"training_time_s: {training_time:.1f}")
    print(f"episodes:        {args.eval_episodes}")
    print(f"fps:             {fps:.1f}")
    print(f"game:            {args.game}")
    print(f"network_size:    {args.network_size}")
    print(f"learning_rate:   {args.lr}")


if __name__ == "__main__":
    main()
