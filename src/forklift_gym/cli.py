"""Small, reproducible SB3 classroom entry point: forklift-lab train/evaluate/frame."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path


def make_env(config, render_mode=None):
    from . import ForkliftEnv, SB3ForkliftEnv
    common = dict(max_steps=config["max_steps"], render_mode=render_mode)
    if config["traffic"] or config["obstacles"]:
        return SB3ForkliftEnv(num_dynamic_agents=config["traffic"],
                             obstacle_count=config["obstacles"],
                             num_boxes=config.get("num_boxes", 4), **common)
    if config.get("randomize", False) or config.get("num_boxes", 4) != 4:
        return ForkliftEnv(num_boxes=config.get("num_boxes", 4), **common)
    return ForkliftEnv(**common)


def vector_env(config, *, statistics=None, render_mode=None):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecNormalize
    def worker(rank):
        def create():
            log_dir = config.get("log_dir") if statistics is None else None
            filename = str(Path(log_dir) / f"worker{rank}") if log_dir else None
            return Monitor(make_env(config, render_mode), filename=filename)
        return create
    cls = SubprocVecEnv if config.get("vec_env") == "subproc" else DummyVecEnv
    base = cls([worker(rank) for rank in range(config["envs"])])
    base.seed(config["seed"])
    if statistics is None:
        normalized = VecNormalize(base, norm_obs=True, norm_reward=True, clip_obs=10.0)
    else:
        normalized = VecNormalize.load(str(statistics), base)
        normalized.training = False
        normalized.norm_reward = False
    wrapped = VecFrameStack(normalized, n_stack=config["history"]) if config["history"] > 1 else normalized
    return wrapped, normalized


def train(args):
    from stable_baselines3 import PPO
    import torch
    torch.set_num_threads(args.threads)
    config = {key: getattr(args, key) for key in
              ("traffic", "obstacles", "max_steps", "envs", "seed", "history", "steps", "rollout", "batch_size", "num_boxes", "randomize", "vec_env", "threads")}
    if args.rollout * args.envs % args.batch_size:
        raise ValueError("envs * rollout must be divisible by batch-size")
    if args.rollout * args.envs < 2 or args.batch_size < 2:
        raise ValueError("PPO requires at least two rollout samples and batch-size >= 2")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("model.zip", "vecnormalize.pkl", "config.json")):
        raise FileExistsError("Choose a new --output directory to preserve the saved run")
    config["log_dir"] = str((output / "monitor").resolve())
    Path(config["log_dir"]).mkdir()
    env, normalized = vector_env(config)
    try:
        model = PPO("MlpPolicy", env, seed=args.seed, device="cpu", verbose=1,
                    n_steps=args.rollout, batch_size=args.batch_size, n_epochs=4,
                    learning_rate=1e-4, gamma=0.995, gae_lambda=0.95,
                    ent_coef=0.01, target_kl=0.03)
        model.learn(total_timesteps=args.steps)
        model.save(str(output / "model"))
        normalized.save(str(output / "vecnormalize.pkl"))
        config["versions"] = {p: importlib.metadata.version(p) for p in
                              ("forklift-gym", "stable-baselines3", "gymnasium", "numpy", "torch")}
        config["actual_timesteps"] = model.num_timesteps
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    finally:
        env.close()
    print(f"Saved model, normalization statistics, and configuration in {output.resolve()}")


def evaluate(args):
    import numpy as np
    from stable_baselines3 import PPO
    run = Path(args.run)
    config = json.loads((run / "config.json").read_text())
    config.update(envs=1, seed=args.seed, vec_env="dummy")
    import torch
    torch.set_num_threads(config.get("threads", 1))
    env, _ = vector_env(config, statistics=run / "vecnormalize.pkl")
    records = []
    try:
        model = PPO.load(str(run / "model.zip"), env=env, device="cpu")
        for episode in range(args.episodes):
            env.seed(args.seed + episode)
            obs = env.reset()
            total, steps, contacts, pickups, deliveries = 0.0, 0, 0, 0, 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, infos = env.step(action)
                info = infos[0]
                total += float(reward[0])
                steps += 1
                pickups += int(info.get("picked_box") is not None)
                deliveries += int(info.get("delivered_box") is not None)
                contacts += int(bool(info.get("collision") or info.get("collided") or info.get("agent_collided")))
                if done[0]:
                    success = bool(info.get("is_success", info.get("distance_to_goal", float("inf")) < 0.3))
                    records.append({"seed": args.seed + episode, "return": total, "steps": steps,
                                    "success": success, "contact_steps": contacts,
                                    "delivered": info.get("delivered_count"), "pickup_events": pickups, "delivery_events": deliveries})
                    break
    finally:
        env.close()
    result = dict(episodes=records, success_rate=float(np.mean([r["success"] for r in records])),
                  mean_return=float(np.mean([r["return"] for r in records])))
    target = run / f"evaluation_seed{args.seed}_n{args.episodes}.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved evaluation: {target.resolve()}")


def frame(args):
    from PIL import Image
    config = dict(traffic=args.traffic, obstacles=args.obstacles, max_steps=args.max_steps, num_boxes=args.num_boxes, randomize=args.randomize)
    env = make_env(config, "rgb_array")
    try:
        obs, _ = env.reset(seed=args.seed)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(env.render()).save(path)
        print(f"Observation: {obs.shape}; action: {env.action_space}; image: {path.resolve()}")
    finally:
        env.close()


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def nonnegative(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("train", train), ("frame", frame)):
        sub = commands.add_parser(name)
        sub.set_defaults(handler=handler)
        sub.add_argument("--num-boxes", type=positive, default=4)
        sub.add_argument("--randomize", action="store_true", help="sample pallet positions at reset")
        sub.add_argument("--traffic", type=nonnegative, default=0)
        sub.add_argument("--obstacles", type=nonnegative, default=0)
        sub.add_argument("--max-steps", type=positive, default=700)
        sub.add_argument("--seed", type=nonnegative, default=0)
        sub.add_argument("--output", required=True)
        if name == "train":
            sub.add_argument("--steps", type=positive, default=100000)
            sub.add_argument("--envs", type=positive, default=4)
            sub.add_argument("--vec-env", choices=("dummy", "subproc"), default="dummy")
            sub.add_argument("--threads", type=positive, default=1)
            sub.add_argument("--history", type=positive, default=4)
            sub.add_argument("--rollout", type=positive, default=512)
            sub.add_argument("--batch-size", type=positive, default=256)
    sub = commands.add_parser("evaluate")
    sub.set_defaults(handler=evaluate)
    sub.add_argument("--run", required=True)
    sub.add_argument("--episodes", type=positive, default=10)
    sub.add_argument("--seed", type=nonnegative, default=10000)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
