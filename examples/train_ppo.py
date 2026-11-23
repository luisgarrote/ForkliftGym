"""Minimal SB3 usage. For normalization/history and saved settings use forklift-lab."""
import argparse
from pathlib import Path
import gymnasium as gym
import forklift_gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default='outputs/simple_ppo')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = Monitor(gym.make('Forklift-v0'), filename=str(output/'episodes'))
    try:
        model = PPO('MlpPolicy', env, verbose=1, device='cpu', seed=args.seed)
        model.learn(total_timesteps=args.steps)
        model.save(str(output/'model'))
    finally:
        env.close()


if __name__ == '__main__':
    main()
