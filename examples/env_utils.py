"""Small shared helpers; training itself is in train_ppo.py using SB3 directly."""
from pathlib import Path
import csv
import json
import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

CASES_PATH = Path(__file__).with_name("evaluation_cases.json")


def load_cases():
    return json.loads(CASES_PATH.read_text())["cases"]


def get_case(case_id):
    return next(case for case in load_cases() if case["id"] == case_id)


def episode(model, env, seed):
    """SB3 VecEnv auto-resets; collect the ending info before the next episode."""
    env.seed(seed)
    obs = env.reset()
    total, length, contacts, pickups, deliveries = 0.0, 0, 0, 0, 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        info = infos[0]
        total += float(rewards[0])
        length += 1
        contacts += int(bool(info.get("collision", info.get("collision_static", False))))
        pickups += int(info.get("picked_box") is not None)
        deliveries += int(info.get("delivered_box") is not None)
        if dones[0]:
            truncated = bool(info.get("TimeLimit.truncated", False))
            return dict(seed=seed, episode_return=total, length=length,
                        success=bool(info.get("is_success", False)),
                        contact_steps=contacts, pallets_delivered=info.get("delivered_count"),
                        pickup_events=pickups, delivery_events=deliveries,
                        terminated=not truncated, truncated=truncated)


def save_results(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for group in sorted({row["group"] for row in rows}):
        cases = [row for row in rows if row["group"] == group]
        returns = [row["episode_return"] for row in cases]
        summary[group] = dict(episodes=len(cases), mean_return=float(np.mean(returns)),
                              std_return=float(np.std(returns)),
                              success_rate=float(np.mean([row["success"] for row in cases])))
    output.with_suffix(".json").write_text(json.dumps(dict(episodes=rows, summary=summary), indent=2)+"\n")
    print(json.dumps(summary, indent=2))
    return summary

from forklift_gym import ForkliftEnv, SB3ForkliftEnv, load_scenario
DEFAULT_HISTORY = 1
POLICY = "MlpPolicy"
VIDEO_FPS = 12.5  # dt = 0.08 s


def make_env(case=None, max_steps=None, traffic=0, obstacles=0, randomize=False):
    if case is not None:
        return SB3ForkliftEnv(scenario=load_scenario(case["scenario"]),
                             max_steps=max_steps or case["max_steps"], render_mode="rgb_array")
    if traffic or obstacles:
        return SB3ForkliftEnv(num_dynamic_agents=traffic, obstacle_count=obstacles,
                             num_boxes=4, max_steps=max_steps or 700, render_mode="rgb_array")
    kwargs = {"num_boxes": 4} if randomize else {}
    return ForkliftEnv(max_steps=max_steps or 700, render_mode="rgb_array", **kwargs)


def prepare_env(base, history=DEFAULT_HISTORY):
    env = DummyVecEnv([lambda: base])
    return VecFrameStack(env, n_stack=history) if history > 1 else env
