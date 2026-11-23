"""Small, deterministic scenario factories for fast robot-learning rollouts.

The generator deliberately produces plain Python/NumPy data and an existing
``Map`` object.  It does not render, build a graph, or run a path planner, so it
is suitable for Colab and for generating a fresh scenario on every reset.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

try:
    from .map_spec import Map, Wall, load_map
except ImportError:  # direct import from the project directory
    from map_spec import Map, Wall, load_map


@dataclass(frozen=True)
class DynamicAgentSpec:
    """Initial state for a non-focus robot in a generated scenario."""

    id: str
    pos: tuple[float, float]
    theta: float
    goal: tuple[float, float]


@dataclass(frozen=True)
class Scenario:
    """Shared scene description consumable by either environment.

    ``map`` contains the common static geometry, spawn, and point goal.
    ``dynamic_agents`` is used by the mobile-robot PettingZoo adapter;
    ``boxes`` and ``goal_area`` are optional forklift task data. Keeping the
    task-specific fields optional lets one scenario definition drive both
    environments without making navigation know about pallets.
    """

    map: Map
    dynamic_agents: tuple[DynamicAgentSpec, ...] = ()
    boxes: Optional[tuple[Any, ...]] = None
    goal_area: Any = None

    def __post_init__(self) -> None:
        # Scenario objects are reused across vectorized rollouts. Copy the
        # mutable task payloads once so an environment cannot mutate the
        # caller's scenario while resetting or stacking boxes.
        object.__setattr__(self, "map", load_map(self.map))
        object.__setattr__(self, "dynamic_agents", _coerce_dynamic_agents(self.dynamic_agents))
        boxes = self.boxes
        meta = getattr(self.map, "meta", {})
        if boxes is None and isinstance(meta, dict):
            boxes = meta.get("boxes")
        if boxes is not None:
            boxes = tuple(copy.deepcopy(item) for item in boxes)
        object.__setattr__(self, "boxes", boxes)
        goal_area = self.goal_area
        if goal_area is None and isinstance(meta, dict):
            goal_area = meta.get("goal_area")
        object.__setattr__(self, "goal_area", copy.deepcopy(goal_area))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly shared scenario payload."""
        dynamic = [
            {
                "id": spec.id,
                "pos": [float(spec.pos[0]), float(spec.pos[1])],
                "theta": float(spec.theta),
                "goal": [float(spec.goal[0]), float(spec.goal[1])],
            }
            for spec in self.dynamic_agents
        ]
        payload: dict[str, Any] = {
            "map": self.map.to_dict(),
            "dynamic_agents": dynamic,
        }
        if self.boxes is not None:
            payload["boxes"] = [
                copy.deepcopy(item.to_dict() if hasattr(item, "to_dict") else item)
                for item in self.boxes
            ]
        if self.goal_area is not None:
            payload["goal_area"] = copy.deepcopy(self.goal_area)
        return payload


def _coerce_dynamic_agents(values: Sequence[Any]) -> tuple[DynamicAgentSpec, ...]:
    """Coerce JSON-friendly dynamic-agent records into shared specs."""
    result: list[DynamicAgentSpec] = []
    for index, value in enumerate(values):
        if isinstance(value, DynamicAgentSpec):
            result.append(value)
            continue
        if not isinstance(value, dict):
            raise TypeError("dynamic_agents entries must be DynamicAgentSpec or dict")
        pos = value.get("pos", value.get("position", [0.0, 0.0]))
        goal = value.get("goal", [0.0, 0.0])
        result.append(DynamicAgentSpec(
            id=str(value.get("id", f"dynamic_{index}")),
            pos=(float(pos[0]), float(pos[1])),
            theta=float(value.get("theta", value.get("orientation", 0.0))),
            goal=(float(goal[0]), float(goal[1])),
        ))
    return tuple(result)


def load_scenario(spec: Any) -> Scenario:
    """Coerce a shared scenario from an object, map, preset, or JSON dict.

    A bare map or map preset becomes a scenario with default task payloads.
    A dictionary may use ``{"map": ..., "boxes": ..., "goal_area": ...}``
    or may be a regular map dictionary whose ``meta`` carries those fields.
    """
    if isinstance(spec, Scenario):
        return spec
    if isinstance(spec, dict) and any(
        key in spec for key in ("map", "scene", "dynamic_agents", "boxes", "goal_area")
    ):
        map_spec = spec.get("map", spec.get("scene"))
        if map_spec is None:
            # Accept a map dictionary with task fields at the top level.
            map_spec = {key: value for key, value in spec.items()
                        if key not in {"dynamic_agents", "boxes", "goal_area"}}
        map_value = load_map(map_spec)
        dynamic = _coerce_dynamic_agents(spec.get("dynamic_agents", ()))
        return Scenario(
            map=map_value,
            dynamic_agents=dynamic,
            boxes=spec.get("boxes"),
            goal_area=spec.get("goal_area"),
        )
    return Scenario(map=load_map(spec), dynamic_agents=())


def make_static_scenario(map_name: str = "empty_arena",
                         dynamic_agents: Sequence[DynamicAgentSpec] = (),
                         *, boxes: Optional[Sequence[Any]] = None,
                         goal_area: Any = None) -> Scenario:
    """Load a fixed scenario with optional mobile traffic or forklift payload."""
    template = load_map(map_name)
    static_map = Map.from_dict(template.to_dict())
    return Scenario(
        map=static_map,
        dynamic_agents=_coerce_dynamic_agents(dynamic_agents),
        boxes=None if boxes is None else tuple(boxes),
        goal_area=goal_area,
    )


def make_random_start_goal_scenario(
    map_name: str = "empty_arena", *, num_dynamic_agents: int = 0,
    robot_radius: float = 0.35, min_start_goal_distance: float = 4.0,
    require_path: bool = True, path_check_resolution: float = 0.2,
) -> "RandomScenario":
    """Keep a preset map fixed while sampling free starts and goals."""
    return RandomScenario(
        base_map=map_name,
        num_dynamic_agents=num_dynamic_agents,
        obstacle_count=0,
        robot_radius=robot_radius,
        min_start_goal_distance=min_start_goal_distance,
        require_path=require_path,
        path_check_resolution=path_check_resolution,
    )


def _sample_patrol_goal(env: Any, rec: Any, *, min_distance: float = 1.0,
                        avoid: Optional[Sequence[float]] = None) -> np.ndarray:
    """Choose a seeded free-space waypoint for a scripted traffic robot.

    A one-shot goal makes background traffic disappear once it reaches its
    destination.  Patrol waypoints are sampled from the environment's seeded
    RNG and checked against the current walls and robot geometry, so the same
    policy keeps moving without changing the observation/action contracts.
    """
    half = float(getattr(env, "world_size", 12.0)) / 2.0
    radius = float(getattr(rec, "radius", 0.0) or getattr(env, "robot_radius", 0.35))
    limit = max(radius + 0.1, half - radius - 0.15)
    current = np.asarray(rec.pos, dtype=np.float32).reshape(2)
    avoided = None if avoid is None else np.asarray(avoid, dtype=np.float32).reshape(2)
    rng = getattr(env, "np_random", None)

    def random_pair() -> np.ndarray:
        if rng is not None and hasattr(rng, "uniform"):
            return np.asarray(rng.uniform(-limit, limit, size=2), dtype=np.float32)
        return np.asarray([
            np.random.uniform(-limit, limit), np.random.uniform(-limit, limit),
        ], dtype=np.float32)

    def circle_hits_wall(point: np.ndarray, wall: Any) -> bool:
        closest_x = min(max(float(point[0]), float(wall.x_lo)), float(wall.x_hi))
        closest_y = min(max(float(point[1]), float(wall.y_lo)), float(wall.y_hi))
        dx = float(point[0]) - closest_x
        dy = float(point[1]) - closest_y
        return dx * dx + dy * dy < (radius + 0.04) ** 2

    def candidate_is_clear(point: np.ndarray) -> bool:
        if float(np.linalg.norm(point - current)) < float(min_distance):
            return False
        if avoided is not None and float(np.linalg.norm(point - avoided)) < 0.75:
            return False
        for wall in getattr(env, "walls", ()):
            if circle_hits_wall(point, wall):
                return False
        # ForkliftEnv has a richer check that includes pallets, forks, the
        # chassis, and other traffic robots.  MobileRobotEnv uses the generic
        # circle checks below because it has no pallet/fork geometry.
        validator = getattr(env, "_agent_position_valid", None)
        if callable(validator):
            try:
                return bool(validator(point, radius, rec))
            except TypeError:
                pass
        focus = getattr(env, "pos", None)
        focus_radius = float(getattr(env, "robot_radius", radius))
        if focus is not None and float(np.linalg.norm(point - np.asarray(focus))) < radius + focus_radius + 0.25:
            return False
        for other in getattr(env, "extras", {}).values():
            if other is rec:
                continue
            other_radius = float(getattr(other, "radius", 0.0) or focus_radius)
            if float(np.linalg.norm(point - np.asarray(other.pos))) < radius + other_radius + 0.25:
                return False
        return True

    # Try a deterministic ring first so patrol routes are stable for a seeded
    # rollout, then fall back to seeded rejection sampling around the map.
    try:
        agent_index = int("".join(ch for ch in str(rec.id) if ch.isdigit()) or 0)
    except ValueError:
        agent_index = 0
    phase = (agent_index * 1.61803398875) % (2.0 * math.pi)
    ring_radius = max(1.5, min(limit * 0.75, max(2.0, half * 0.55)))
    for index in range(12):
        angle = phase + index * (2.0 * math.pi / 12.0)
        point = np.asarray([
            math.cos(angle) * ring_radius,
            math.sin(angle) * ring_radius,
        ], dtype=np.float32)
        if candidate_is_clear(point):
            return point
    for _ in range(80):
        point = random_pair()
        if candidate_is_clear(point):
            return point
    # Extremely dense scenes can exhaust rejection sampling.  The reflected
    # goal stays inside the arena and still gives the controller a new heading.
    fallback = np.clip(-np.asarray(rec.goal, dtype=np.float32), -limit, limit)
    return fallback if float(np.linalg.norm(fallback - current)) >= 0.25 else np.clip(
        current + np.asarray([0.8, 0.6], dtype=np.float32), -limit, limit,
    )


def goal_seek_policy(agent_id: str, speed: float = 0.85,
                     turn_gain: float = 1.2):
    """Return a scripted policy that continuously patrols free-space waypoints."""
    agent_id = str(agent_id)

    def policy(_obs, env):
        rec = env.get_agent(agent_id)
        try:
            agent_index = int("".join(ch for ch in str(rec.id) if ch.isdigit()) or 0)
        except ValueError:
            agent_index = 0
        stuck_steps = int(getattr(rec, "stuck_steps", 0))
        if stuck_steps > 0:
            # Turn in place for several frames after contact.  A single tiny
            # turn still points into a shelf; a short recovery window lets the
            # robot leave the obstacle before it resumes forward motion.
            rec.stuck_steps = stuck_steps - 1
            turn_direction = 1.0 if agent_index % 2 == 0 else -1.0
            return np.array([0.0, turn_direction], dtype=np.float32)
        dx = float(rec.goal[0] - rec.pos[0])
        dy = float(rec.goal[1] - rec.pos[1])
        distance = math.hypot(dx, dy)
        if distance < 0.3 or bool(getattr(rec, "last_collided", False)):
            was_blocked = bool(getattr(rec, "last_collided", False))
            old_goal = rec.goal.copy()
            if was_blocked:
                rec.stuck_steps = 6
            rec.goal[:] = _sample_patrol_goal(
                env, rec, avoid=old_goal if was_blocked else None,
            )
            dx = float(rec.goal[0] - rec.pos[0])
            dy = float(rec.goal[1] - rec.pos[1])
            distance = math.hypot(dx, dy)
            # A degenerate fallback can still land directly on the current
            # point; turn in place for one step so the next waypoint differs.
            if distance < 1e-4:
                return np.array([0.0, 0.65], dtype=np.float32)
            if was_blocked:
                rec.stuck_steps = max(0, int(rec.stuck_steps) - 1)
                turn_direction = 1.0 if agent_index % 2 == 0 else -1.0
                return np.array([0.0, turn_direction], dtype=np.float32)
        desired = math.atan2(dy, dx)
        error = (desired - float(rec.theta) + math.pi) % (2.0 * math.pi) - math.pi
        linear = float(speed) * math.exp(-abs(error) / max(1e-6, float(turn_gain)))
        angular = float(np.clip(error / (math.pi / 2.0), -1.0, 1.0))
        return np.array([np.clip(linear, -1.0, 1.0), angular], dtype=np.float32)

    return policy


class RandomScenario:
    """Generate open or lightly-obstructed navigation scenes.

    Parameters are intentionally conservative: small AABB obstacles are
    sampled without overlap, while all robot and goal points are rejection-
    sampled in free space.  ``obstacle_count=0`` is the fastest mode and is a
    useful curriculum starting point; a few obstacles add variety without
    requiring a heavyweight simulator. By default, a small inflated occupancy
    grid is flood-filled so disconnected start/goal pairs are resampled; this
    checks reachability without computing an optimal path.
    """

    def __init__(self, world_size: float = 12.0, num_dynamic_agents: int = 2,
                 obstacle_count: int = 0, robot_radius: float = 0.35,
                 base_map=None, obstacle_size: tuple[float, float] = (0.7, 1.5),
                 min_start_goal_distance: float = 4.0,
                 require_path: bool = True, path_check_resolution: float = 0.2,
                 max_resample_attempts: int = 100) -> None:
        self.world_size = float(world_size)
        self.num_dynamic_agents = int(num_dynamic_agents)
        self.obstacle_count = int(obstacle_count)
        self.robot_radius = float(robot_radius)
        self.obstacle_size = (float(obstacle_size[0]), float(obstacle_size[1]))
        self.min_start_goal_distance = float(min_start_goal_distance)
        self.require_path = bool(require_path)
        self.path_check_resolution = float(path_check_resolution)
        self.max_resample_attempts = int(max_resample_attempts)
        if self.world_size <= 2.0 * self.robot_radius + 0.2:
            raise ValueError("world_size is too small for the robot radius")
        if self.num_dynamic_agents < 0 or self.obstacle_count < 0:
            raise ValueError("agent and obstacle counts must be non-negative")
        if self.obstacle_size[0] <= 0.0 or self.obstacle_size[1] < self.obstacle_size[0]:
            raise ValueError("obstacle_size must be (positive_min, max >= min)")
        if self.min_start_goal_distance < 0.0:
            raise ValueError("min_start_goal_distance must be non-negative")
        if self.path_check_resolution <= 0.0:
            raise ValueError("path_check_resolution must be positive")
        if self.max_resample_attempts < 1:
            raise ValueError("max_resample_attempts must be at least 1")
        self.base_map = load_map(base_map) if base_map is not None else None
        self._rng = np.random.default_rng()

    def sample(self, seed: Optional[int] = None) -> Scenario:
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        attempts = self.max_resample_attempts if self.require_path else 1
        for _ in range(attempts):
            scene = self._sample_once(rng)
            if not self.require_path or self._has_free_path(scene):
                return scene
        raise RuntimeError(
            "could not sample a scenario with a connected start/goal; "
            "reduce obstacles or the minimum start/goal distance"
        )

    def _sample_once(self, rng) -> Scenario:
        if self.base_map is None:
            world_size = self.world_size
            walls: list[Wall] = []
            elements = []
            template_meta: dict[str, Any] = {}
        else:
            # A fresh serializable copy prevents one rollout from mutating the
            # map used as the next rollout's template.
            template = Map.from_dict(self.base_map.to_dict())
            world_size = float(template.world_size)
            walls = list(template.collision_walls())
            elements = list(template.elements)
            template_meta = dict(template.meta)

        half = world_size / 2.0
        if self.base_map is None:
            walls = self._sample_obstacles(rng, walls, half)

        spawn = self._sample_point(rng, walls, half, self.robot_radius)
        goal = self._sample_point(
            rng, walls, half, self.robot_radius,
            forbidden=(spawn,), min_distance=self.min_start_goal_distance,
        )
        theta = float(rng.uniform(-math.pi, math.pi))
        scene = Map(
            name="random" if self.base_map is None else str(self.base_map.name),
            world_size=world_size,
            walls=walls if self.base_map is None else list(self.base_map.walls),
            spawn=[float(spawn[0]), float(spawn[1]), theta],
            goal=[float(goal[0]), float(goal[1])],
            elements=elements,
            meta={
                **template_meta,
                "generated": True,
                "obstacle_count": self.obstacle_count,
                "path_checked": self.require_path,
                "path_check_resolution": self.path_check_resolution,
            },
        )

        occupied = [spawn, goal]
        specs = []
        for i in range(self.num_dynamic_agents):
            pos = self._sample_point(rng, walls, half, self.robot_radius,
                                     forbidden=tuple(occupied), min_distance=1.0)
            agent_goal = self._sample_point(
                rng, walls, half, self.robot_radius,
                forbidden=tuple(occupied), min_distance=1.0,
            )
            occupied.append(pos)
            specs.append(DynamicAgentSpec(
                id=f"dynamic_{i}",
                pos=(float(pos[0]), float(pos[1])),
                theta=float(rng.uniform(-math.pi, math.pi)),
                goal=(float(agent_goal[0]), float(agent_goal[1])),
            ))
        return Scenario(map=scene, dynamic_agents=tuple(specs))

    def _has_free_path(self, scene: Scenario) -> bool:
        """Fast conservative connectivity test; this is not an optimal planner."""
        world_size = float(scene.map.world_size)
        half = world_size / 2.0
        resolution = self.path_check_resolution
        n = max(2, int(math.ceil(world_size / resolution)))
        cell = world_size / n
        axis = -half + (np.arange(n, dtype=np.float32) + 0.5) * cell
        xx, yy = np.meshgrid(axis, axis, indexing="xy")

        # Inflate every obstacle by the robot radius plus a small numerical
        # margin. Boundary cells are blocked for the same reason.
        clearance = float(self.robot_radius) + 0.05
        blocked = (np.abs(xx) > half - clearance) | (np.abs(yy) > half - clearance)
        for wall in scene.map.collision_walls():
            cx = np.clip(xx, wall.x_lo, wall.x_hi)
            cy = np.clip(yy, wall.y_lo, wall.y_hi)
            blocked |= ((xx - cx) ** 2 + (yy - cy) ** 2) < clearance ** 2

        def cell_index(point):
            ix = int(np.clip((float(point[0]) + half) / cell, 0, n - 1))
            iy = int(np.clip((float(point[1]) + half) / cell, 0, n - 1))
            return iy, ix

        start = cell_index(scene.map.spawn[:2])
        goal = cell_index(scene.map.goal)
        if blocked[start] or blocked[goal]:
            return False

        visited = np.zeros((n, n), dtype=np.bool_)
        visited[start] = True
        queue = deque([start])
        while queue:
            y, x = queue.popleft()
            if (y, x) == goal:
                return True
            for ny, nx in ((y - 1, x), (y + 1, x),
                           (y, x - 1), (y, x + 1)):
                if 0 <= ny < n and 0 <= nx < n:
                    if not blocked[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        return False

    def _sample_obstacles(self, rng, walls: list[Wall], half: float) -> list[Wall]:
        if self.obstacle_count == 0:
            return walls
        out = list(walls)
        margin = self.robot_radius + 0.25
        for _ in range(self.obstacle_count):
            for _attempt in range(100):
                width = float(rng.uniform(self.obstacle_size[0], self.obstacle_size[1]))
                height = float(rng.uniform(self.obstacle_size[0], self.obstacle_size[1]))
                x = float(rng.uniform(-half + margin + width / 2.0,
                                      half - margin - width / 2.0))
                y = float(rng.uniform(-half + margin + height / 2.0,
                                      half - margin - height / 2.0))
                candidate = Wall(x - width / 2.0, y - height / 2.0,
                                 x + width / 2.0, y + height / 2.0)
                if all(not self._boxes_near(candidate, old, margin)
                       for old in out):
                    out.append(candidate)
                    break
        return out

    @staticmethod
    def _boxes_near(a: Wall, b: Wall, margin: float) -> bool:
        return not (a.x_hi + margin < b.x_lo or b.x_hi + margin < a.x_lo or
                    a.y_hi + margin < b.y_lo or b.y_hi + margin < a.y_lo)

    def _sample_point(self, rng, walls: Sequence[Wall], half: float,
                      radius: float, forbidden=(), min_distance: float = 0.0):
        limit = half - radius - 0.1
        for _ in range(1000):
            point = np.array([rng.uniform(-limit, limit),
                              rng.uniform(-limit, limit)], dtype=np.float32)
            if any(self._circle_hits_wall(point, radius, wall) for wall in walls):
                continue
            if any(float(np.linalg.norm(point - other)) < min_distance
                   for other in forbidden):
                continue
            return point
        raise RuntimeError("could not sample a free point; reduce obstacles or agent count")

    @staticmethod
    def _circle_hits_wall(point, radius: float, wall: Wall) -> bool:
        cx = min(max(float(point[0]), wall.x_lo), wall.x_hi)
        cy = min(max(float(point[1]), wall.y_lo), wall.y_hi)
        return ((float(point[0]) - cx) ** 2 + (float(point[1]) - cy) ** 2) < radius ** 2


class CurriculumScenario:
    """Random scenario generator whose object count ramps to a maximum.

    ``progress`` is controlled by the training loop in ``[0, 1]``. Obstacles
    Obstacles are increased first, then dynamic agents are introduced. A
    fraction of later samples replays easier stages. The standard learner
    observes traffic through LIDAR, so its observation shape stays fixed even
    as the scene gains more robots.
    """

    def __init__(self, world_size: float = 12.0, *,
                 max_dynamic_agents: int = 0, max_obstacles: int = 0,
                 robot_radius: float = 0.35, base_map=None,
                 obstacle_size: tuple[float, float] = (0.7, 1.5),
                 min_start_goal_distance: float = 4.0,
                 require_path: bool = True, path_check_resolution: float = 0.2,
                 max_resample_attempts: int = 100,
                 replay_probability: float = 0.25,
                 obstacle_ramp_end: float = 0.4,
                 agent_ramp_end: float = 0.7) -> None:
        self.base_map = load_map(base_map) if base_map is not None else None
        if self.base_map is not None:
            world_size = float(self.base_map.world_size)
        self.world_size = float(world_size)
        self.max_dynamic_agents = int(max_dynamic_agents)
        self.max_obstacles = int(max_obstacles)
        if self.max_dynamic_agents < 0 or self.max_obstacles < 0:
            raise ValueError("curriculum object counts must be non-negative")
        if self.base_map is not None and self.max_obstacles:
            raise ValueError(
                "a fixed base_map cannot receive random obstacles; use a random map"
            )
        self.robot_radius = float(robot_radius)
        self.obstacle_size = (float(obstacle_size[0]), float(obstacle_size[1]))
        self.min_start_goal_distance = float(min_start_goal_distance)
        self.require_path = bool(require_path)
        self.path_check_resolution = float(path_check_resolution)
        self.max_resample_attempts = int(max_resample_attempts)
        self.replay_probability = float(replay_probability)
        self.obstacle_ramp_end = float(obstacle_ramp_end)
        self.agent_ramp_end = float(agent_ramp_end)
        if not 0.0 <= self.replay_probability <= 1.0:
            raise ValueError("replay_probability must be in [0, 1]")
        if not 0.0 < self.obstacle_ramp_end < self.agent_ramp_end <= 1.0:
            raise ValueError(
                "ramp endpoints must satisfy 0 < obstacle_ramp_end "
                "< agent_ramp_end <= 1"
            )
        # These public maximums let wrappers know the requested training range;
        # the standard learner still uses a fixed LIDAR-only observation.
        self.num_dynamic_agents = self.max_dynamic_agents
        self.obstacle_count = self.max_obstacles
        self._progress = 0.0

    @property
    def progress(self) -> float:
        return self._progress

    def set_progress(self, progress: float) -> dict:
        """Set curriculum progress and return the active object counts."""
        progress = float(progress)
        if not math.isfinite(progress):
            raise ValueError("curriculum progress must be finite")
        self._progress = float(np.clip(progress, 0.0, 1.0))
        return self.state()

    def state(self) -> dict:
        """Return the active curriculum stage for logging and tests."""
        return self._state_for_progress(self._progress)

    def sample(self, seed: Optional[int] = None) -> Scenario:
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        sample_progress = self._progress
        # Revisit an easier stage during later training so PPO does not forget
        # the navigation skill learned before the next object is introduced.
        if sample_progress > 0.0 and rng.random() < self.replay_probability:
            sample_progress = float(rng.uniform(0.0, sample_progress))
        active = self._state_for_progress(sample_progress)
        scene_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        generator = RandomScenario(
            world_size=self.world_size,
            num_dynamic_agents=active["dynamic_agents"],
            obstacle_count=active["obstacles"],
            robot_radius=self.robot_radius,
            base_map=self.base_map,
            obstacle_size=self.obstacle_size,
            min_start_goal_distance=self.min_start_goal_distance,
            require_path=self.require_path,
            path_check_resolution=self.path_check_resolution,
            max_resample_attempts=self.max_resample_attempts,
        )
        scene = generator.sample(seed=scene_seed)
        scene.map.meta.update({
            "curriculum_progress": float(self._progress),
            "curriculum_sample_progress": sample_progress,
            "curriculum_dynamic_agents": active["dynamic_agents"],
            "curriculum_obstacles": active["obstacles"],
        })
        return scene

    def _state_for_progress(self, progress: float) -> dict:
        obstacle_progress = self._phase_progress(
            progress, 0.0, self.obstacle_ramp_end
        )
        agent_progress = self._phase_progress(
            progress, self.obstacle_ramp_end, self.agent_ramp_end
        )
        return {
            "progress": float(progress),
            "dynamic_agents": self._count_at(self.max_dynamic_agents, agent_progress),
            "obstacles": self._count_at(self.max_obstacles, obstacle_progress),
        }

    @staticmethod
    def _phase_progress(progress: float, start: float, end: float) -> float:
        return float(np.clip((progress - start) / max(1e-9, end - start), 0.0, 1.0))

    @staticmethod
    def _count_at(maximum: int, progress: float) -> int:
        if maximum <= 0:
            return 0
        # The extra interval keeps the zero-object stage visible at the start
        # and reaches the maximum before/at the end of the schedule.
        return min(maximum, int(math.floor(progress * (maximum + 1))))


__all__ = [
    "DynamicAgentSpec", "Scenario", "load_scenario", "RandomScenario", "CurriculumScenario",
    "goal_seek_policy",
    "make_static_scenario", "make_random_start_goal_scenario",
]
