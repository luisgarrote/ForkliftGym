"""PettingZoo parallel adapter for the forklift task.

The forklift remains the only learning agent (``main``).  Other records in a
shared :class:`Scenario` are scripted traffic robots: they move in the same
forklift physics, are sensed by the forklift lidar, collide with the forklift
and pallets, and are rendered with the common shadow style.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import gymnasium as gym
import numpy as np

try:
    from pettingzoo import ParallelEnv
    _PETTINGZOO_IMPORT_ERROR = None
except ImportError as exc:  # keep Gymnasium-only installs importable
    _PETTINGZOO_IMPORT_ERROR = exc

    class ParallelEnv:  # type: ignore[no-redef]
        pass

try:
    from .forklift_env import ForkliftEnv
    from .scenarios import RandomScenario, Scenario, goal_seek_policy, load_scenario
except ImportError:  # direct import from the project directory
    from forklift_env import ForkliftEnv
    from scenarios import RandomScenario, Scenario, goal_seek_policy, load_scenario


class ForkliftParallelEnv(ParallelEnv):
    """One learner-controlled forklift navigating among scripted traffic."""

    metadata = {
        "name": "forklift_parallel_v0",
        "render_modes": ["human", "rgb_array"],
        "render_fps": ForkliftEnv.metadata["render_fps"],
        "is_parallelizable": True,
    }

    def __init__(self, *, scenario_generator: Optional[RandomScenario] = None,
                 scenario: Optional[Scenario] = None, world_size: float = 12.0,
                 num_dynamic_agents: int = 2, obstacle_count: int = 0,
                 robot_radius: float = 0.45, max_steps: int = 700,
                 boxes: Optional[Any] = None, num_boxes: Optional[int] = None,
                 goal_area: Optional[Any] = None,
                 shadow_offset: Sequence[float] = (0.10, -0.10),
                 dynamic_policies: Optional[Mapping[str, Callable]] = None,
                 render_mode: Optional[str] = None, **forklift_kwargs) -> None:
        if _PETTINGZOO_IMPORT_ERROR is not None:
            raise ImportError(
                "ForkliftParallelEnv requires pettingzoo; install requirements.txt"
            ) from _PETTINGZOO_IMPORT_ERROR
        if scenario is not None and scenario_generator is not None:
            raise ValueError("pass either scenario or scenario_generator, not both")
        if scenario is not None:
            scenario = load_scenario(scenario)
        if scenario is None and scenario_generator is None:
            scenario_generator = RandomScenario(
                world_size=world_size,
                num_dynamic_agents=num_dynamic_agents,
                obstacle_count=obstacle_count,
                robot_radius=robot_radius,
            )
        if scenario is not None:
            world_size = float(scenario.map.world_size)

        self.scenario_generator = scenario_generator
        self._fixed_scenario = scenario
        self.dynamic_policies = dict(dynamic_policies or {})
        self.possible_agents = ["main"]
        self.agents: list[str] = []
        self._last_scenario: Optional[Scenario] = None
        self._episode_steps = 0

        env_kwargs = dict(forklift_kwargs)
        env_kwargs.update({
            "world_size": world_size,
            "robot_radius": robot_radius,
            "max_steps": max_steps,
            "boxes": boxes,
            "num_boxes": num_boxes,
            "goal_area": goal_area,
            "shadow_offset": shadow_offset,
            "render_mode": render_mode,
        })
        if scenario is not None:
            env_kwargs["scenario"] = scenario
        self.env = ForkliftEnv(**env_kwargs)
        self._action_space = self.env.action_space
        self._observation_space = self.env.observation_space

    def observation_space(self, agent):
        self._check_agent(agent)
        return self._observation_space

    def action_space(self, agent):
        self._check_agent(agent)
        return self._action_space

    def _apply_scenario(self, sampled: Scenario) -> None:
        """Swap the shared map/task data before the base reset."""
        self.env._map_override = sampled.map
        self.env._map_input = sampled.map
        self.env.world_size = float(sampled.map.world_size)
        self.env._spawn_spec = self.env._coerce_spawn(sampled.map.spawn)

        goal_value = sampled.goal_area if sampled.goal_area is not None else sampled.map.goal
        self.env._goal_center_spec, self.env._goal_size_spec = self.env._coerce_goal_area(goal_value)

        # Rebuild the static wall list for a generated map.  Movable boxes are
        # represented separately in the shared scenario; other colliding map
        # objects remain static obstacles.
        self.env._base_walls = list(sampled.map.walls)
        box_labels: set[str] = set()
        for element in getattr(sampled.map, "elements", []) or []:
            label = str(getattr(element, "label", "") or "")
            is_box = str(getattr(element, "kind", "")).lower() == "object" and any(
                word in label.lower() for word in ("box", "crate", "pallet", "cargo")
            )
            if is_box:
                box_labels.add(label)
                continue
            aabb = getattr(element, "aabb", None)
            if getattr(element, "collide", False) and aabb is not None:
                self.env._base_walls.append(aabb)

        if sampled.boxes is not None:
            self.env._box_specs = self.env._normalize_box_specs(sampled.boxes)
            self.env._num_boxes = None
        elif box_labels:
            map_boxes = []
            for index, element in enumerate(getattr(sampled.map, "elements", []) or []):
                label = str(getattr(element, "label", "") or "")
                if label not in box_labels:
                    continue
                map_boxes.append({
                    "id": label or f"map_box_{index}",
                    "pos": [float(element.x), float(element.y)],
                    "width": float(element.w),
                    "depth": float(element.h),
                    "height": 0.6,
                })
            if map_boxes:
                self.env._box_specs = self.env._normalize_box_specs(map_boxes)
                self.env._num_boxes = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        del options
        # Auto-resets must also be reproducible after an explicitly seeded reset.
        # Use the base environment RNG instead of an unseeded scenario RNG.
        if seed is None:
            seed = int(self.env.np_random.integers(0, 2**31 - 1))
        if self._fixed_scenario is not None:
            sampled = self._fixed_scenario
        else:
            sampled = self.scenario_generator.sample(seed=seed)
        self._last_scenario = sampled

        self.env.clear_agents()
        self._apply_scenario(sampled)
        observation, info = self.env.reset(seed=seed)
        for spec in sampled.dynamic_agents:
            method = self.dynamic_policies.get(spec.id)
            if method is None:
                method = goal_seek_policy(spec.id)
            self.env.add_agent(
                spec.id,
                pos=np.asarray(spec.pos, dtype=np.float32),
                theta=spec.theta,
                goal=np.asarray(spec.goal, dtype=np.float32),
                method=method,
                controllable=True,
            )

        # Traffic is registered after the base reset so it can use the reset
        # scene's box and wall state.  Rebuild the first observation now that
        # the forklift lidar must see those newly registered agents.
        observation = self.env._get_obs()
        info = self.env._info()

        self._episode_steps = 0
        self.agents = list(self.possible_agents)
        info = dict(info)
        info.update({
            "scenario": sampled.map.name,
            "num_dynamic_agents": len(sampled.dynamic_agents),
        })
        return {"main": observation}, {"main": info}

    def step(self, actions):
        if not self.agents:
            raise RuntimeError("step() called after the episode ended; call reset()")
        if not isinstance(actions, Mapping):
            raise TypeError("PettingZoo ParallelEnv.step expects an action dict")

        active_agents = list(self.agents)
        main_action = actions.get("main", np.zeros(3, dtype=np.float32))
        observation, reward, terminated, truncated, info = self.env.step(main_action)
        self._episode_steps += 1
        info = dict(info)
        info["dynamic_agents"] = {
            agent_id: {
                "pos": entry["pos"].copy(),
                "theta": entry["theta"],
                "collided": entry["collided"],
            }
            for agent_id, entry in info.get("agents", {}).items()
        }

        observations = {agent: observation for agent in active_agents}
        rewards = {agent: float(reward) for agent in active_agents}
        terminations = {agent: bool(terminated) for agent in active_agents}
        truncations = {agent: bool(truncated) for agent in active_agents}
        infos = {agent: dict(info) for agent in active_agents}
        if terminated or truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def set_curriculum_progress(self, progress: float):
        if self.scenario_generator is None or not hasattr(self.scenario_generator, "set_progress"):
            raise RuntimeError("this environment was not created with a curriculum generator")
        return self.scenario_generator.set_progress(progress)

    def state(self) -> np.ndarray:
        """Return forklift and traffic state for centralized critics/logging."""
        values = [
            float(self.env.pos[0]), float(self.env.pos[1]), float(self.env.theta),
            float(self.env.goal[0]), float(self.env.goal[1]),
            float(self.env.fork_height), float(self.env.delivered_count),
            float(self.env.remaining_count),
        ]
        for agent_id in sorted(self.env.extras):
            rec = self.env.extras[agent_id]
            values.extend([
                float(rec.pos[0]), float(rec.pos[1]), float(rec.theta),
                float(rec.vel[0]),
            ])
        return np.asarray(values, dtype=np.float32)

    def _check_agent(self, agent):
        if agent not in self.possible_agents:
            raise KeyError(f"unknown learning agent {agent!r}")


class SB3ForkliftEnv(gym.Env):
    """Gymnasium single-agent view over :class:`ForkliftParallelEnv`.

    Stable-Baselines3 consumes one action and one observation per environment
    slot.  This thin adapter keeps the PettingZoo traffic semantics intact:
    PPO controls ``main`` while scripted agents are advanced by the parallel
    environment on every step.
    """

    metadata = ForkliftParallelEnv.metadata

    def __init__(self, **parallel_kwargs) -> None:
        super().__init__()
        self.parallel_env = ForkliftParallelEnv(**parallel_kwargs)
        self.action_space = self.parallel_env.action_space("main")
        self.observation_space = self.parallel_env.observation_space("main")

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        observations, infos = self.parallel_env.reset(seed=seed, options=options)
        return observations["main"], infos["main"]

    def step(self, action):
        observations, rewards, terminations, truncations, infos = self.parallel_env.step({
            "main": np.asarray(action, dtype=np.float32),
        })
        return (
            observations["main"],
            float(rewards["main"]),
            bool(terminations["main"]),
            bool(truncations["main"]),
            infos["main"],
        )

    def render(self):
        return self.parallel_env.render()

    def close(self):
        self.parallel_env.close()

    @property
    def forklift(self) -> ForkliftEnv:
        """Expose the underlying renderer/state for evaluation helpers."""
        return self.parallel_env.env


__all__ = ["ForkliftParallelEnv", "SB3ForkliftEnv"]
