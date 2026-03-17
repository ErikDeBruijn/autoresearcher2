#!/usr/bin/env python3
"""Record Atari gameplay video from a trained model.

Standalone script that runs AFTER train_atari.py finishes.
Loads the saved model, records episodes, and outputs an artifact_video line
that the shell executor picks up.

Expected model path: /tmp/atari_model_latest.zip (SB3 format)
Output: /tmp/atari_gameplay.mp4

Usage:
    python record_video.py [--model /tmp/atari_model_latest.zip] [--output /tmp/atari_gameplay.mp4]
"""
import argparse
import os
import sys

DEFAULT_MODEL_PATH = "/tmp/atari_model_latest.zip"
DEFAULT_OUTPUT_PATH = "/tmp/atari_gameplay.mp4"


def record(model_path: str, output_path: str, episodes: int = 3, max_steps: int = 5000):
    if not os.path.exists(model_path):
        print(f"record_video: model not found at {model_path}, skipping video recording",
              file=sys.stderr)
        return False

    try:
        import imageio
        import numpy as np
        from stable_baselines3 import PPO
        from stable_baselines3.common.atari_wrappers import AtariWrapper
        import gymnasium as gym
    except ImportError as e:
        print(f"record_video: missing dependency ({e}), skipping video recording",
              file=sys.stderr)
        return False

    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"record_video: failed to load model ({e}), skipping video recording",
              file=sys.stderr)
        return False

    # Create environment matching typical Atari setup
    env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")
    env = AtariWrapper(env)

    frames = []
    total_reward = 0.0

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            # Capture frame from unwrapped env for full resolution
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            if terminated or truncated:
                break

        total_reward += ep_reward
        print(f"record_video: episode {ep+1}/{episodes} reward={ep_reward:.1f}")

    env.close()

    if not frames:
        print("record_video: no frames captured, skipping", file=sys.stderr)
        return False

    # Write video
    writer = imageio.get_writer(output_path, fps=30)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    avg_reward = total_reward / episodes
    print(f"record_video: saved {len(frames)} frames to {output_path} "
          f"(avg_reward={avg_reward:.1f})")
    print(f"artifact_video: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Record Atari gameplay video")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                        help=f"Path to saved model (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH,
                        help=f"Output video path (default: {DEFAULT_OUTPUT_PATH})")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to record (default: 3)")
    args = parser.parse_args()

    success = record(args.model, args.output, episodes=args.episodes)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
