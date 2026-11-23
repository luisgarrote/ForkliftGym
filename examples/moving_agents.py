"""Render genuine moving traffic while the controlled forklift waits."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from forklift_gym import SB3ForkliftEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output', default='outputs/traffic.gif')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    env = SB3ForkliftEnv(num_dynamic_agents=2, obstacle_count=2,
                         num_boxes=4, render_mode='rgb_array', max_steps=args.steps+1)
    frames = []
    try:
        env.reset(seed=args.seed)
        for step in range(args.steps+1):
            if step % 4 == 0 or step == args.steps:
                frame = Image.fromarray(env.render())
                frame.thumbnail((640,640))
                frames.append(frame)
            if step < args.steps:
                _, _, terminated, truncated, _ = env.step(np.zeros(3, dtype=np.float32))
                if terminated or truncated:
                    break
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=320, loop=0)
        print(path)
    finally:
        env.close()


if __name__ == '__main__':
    main()
