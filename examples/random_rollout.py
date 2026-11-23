"""Run native Gymnasium controls and save an actual environment RGB frame."""
import argparse
from pathlib import Path
import gymnasium as gym
import forklift_gym  # registers Forklift-v0 and ForkliftTraffic-v0
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traffic', type=int, default=0)
    parser.add_argument('--obstacles', type=int, default=0)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output', default='outputs/rollout.png')
    args = parser.parse_args()
    if min(args.traffic, args.obstacles, args.steps) < 0:
        parser.error('counts must be nonnegative')
    if args.traffic or args.obstacles:
        env = gym.make('ForkliftTraffic-v0', num_dynamic_agents=args.traffic,
                       obstacle_count=args.obstacles, num_boxes=4, render_mode='rgb_array')
    else:
        env = gym.make('Forklift-v0', render_mode='rgb_array')
    try:
        obs, info = env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
        print('Observation:', obs.shape, 'Action:', env.action_space)
        for _ in range(args.steps):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                break
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(env.render()).save(output)
        print(f'Delivered {info["delivered_count"]}/4; frame saved to {output}')
    finally:
        env.close()


if __name__ == '__main__':
    main()
