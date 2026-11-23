"""A small forklift and warehouse-box Gymnasium environment.

The original project contains a differential-drive navigation task.  This
module keeps that task intact and adds a goal-conditioned material-handling
task on the same 2-D, top-down world model:

* the action is ``[drive, steer, lift]``;
* the forklift automatically picks the top box of a stack when its low forks
  make contact; the lift control raises and lowers the load;
* a picked box stays attached until it is lowered in the goal area;
* after a delivery, the goal changes to the closest accessible box;
* boxes at one position form a real stack, and delivered boxes form a stack in
  the goal area too.

The environment intentionally has no dependency on the mobile-robot step
implementation.  It shares the project's ``Wall`` and map format, which
keeps maps and rendering conventions interoperable without changing the
existing API.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .primitives import AgentRecord, Wall, _ray_aabb_t, get_wall_hatch
from .map_spec import load_map
from .scenarios import Scenario, load_scenario


@dataclass(eq=False)
class ForkliftBox:
    """Mutable state for one box.

    ``pos`` is the horizontal centre in world coordinates and ``theta`` is the
    heading of its oriented rectangular footprint.  ``stack_level`` is zero
    for the bottom box at its source stack.  Once delivered, the box keeps its
    source level for bookkeeping and receives a ``goal_level`` for its
    destination stack.
    """

    id: str
    pos: np.ndarray
    width: float = 0.8
    depth: float = 0.8
    height: float = 0.6
    stack_level: int = 0
    source_pos: Optional[np.ndarray] = None
    picked: bool = False
    carried: bool = False
    delivered: bool = False
    goal_level: int = -1
    z: float = 0.0
    # Heading of the pallet footprint in the BEV plane.  Keeping this field at
    # the end preserves the older positional constructor signature.
    theta: float = 0.0

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.pos = np.asarray(self.pos, dtype=np.float32).reshape(2).copy()
        if self.source_pos is None:
            self.source_pos = self.pos.copy()
        else:
            self.source_pos = np.asarray(self.source_pos, dtype=np.float32).reshape(2).copy()
        self.width = float(self.width)
        self.depth = float(self.depth)
        self.height = float(self.height)
        self.stack_level = int(self.stack_level)
        self.goal_level = int(self.goal_level)
        self.picked = bool(self.picked)
        self.carried = bool(self.carried)
        self.delivered = bool(self.delivered)
        self.z = float(self.z)
        self.theta = (float(self.theta) + math.pi) % (2.0 * math.pi) - math.pi
        if self.width <= 0.0 or self.depth <= 0.0 or self.height <= 0.0:
            raise ValueError("box width, depth, and height must be positive")

    @property
    def position(self) -> np.ndarray:
        """Alias for callers that use ``position`` instead of ``pos``."""
        return self.pos

    @position.setter
    def position(self, value: Sequence[float]) -> None:
        self.pos = np.asarray(value, dtype=np.float32).reshape(2).copy()
        if not self.carried and not self.delivered:
            self.source_pos = self.pos.copy()

    @property
    def stack_index(self) -> int:
        return self.stack_level

    @stack_index.setter
    def stack_index(self, value: int) -> None:
        self.stack_level = int(value)

    @property
    def status(self) -> str:
        if self.delivered:
            return "delivered"
        if self.carried:
            return "carried"
        return "available"

    def copy(self) -> "ForkliftBox":
        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "pos": [float(self.pos[0]), float(self.pos[1])],
            "width": self.width,
            "depth": self.depth,
            "height": self.height,
            "stack_level": self.stack_level,
            "source_pos": [float(self.source_pos[0]), float(self.source_pos[1])],
            "picked": self.picked,
            "carried": self.carried,
            "delivered": self.delivered,
            "goal_level": self.goal_level,
            "z": self.z,
            "theta": float(self.theta),
        }


class ForkliftEnv(gym.Env):
    """A top-down forklift that collects and delivers stacked boxes.

    Parameters are deliberately plain Python values so the environment works
    with ``gym.make`` and with the map editor's JSON files.  ``boxes`` accepts
    :class:`ForkliftBox` instances, dictionaries, ``[x, y]`` pairs, or
    ``(x, y)`` pairs.  Repeating the same position creates a stack; the order
    in the input is normalized into bottom-to-top levels.

    The current goal is exposed as ``env.goal`` and ``info["goal"]``.  While
    empty-handed it is the nearest accessible box.  While carrying, it is the
    centre of the goal area.  Delivered boxes remain in a destination stack,
    so no box can be silently abandoned after pickup.

    A shared ``Scenario`` can supply the map, pallet specs, and goal area so
    the same scene definition can also be consumed by the mobile-robot task.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        world_size: float = 12.0,
        robot_radius: float = 0.45,
        max_linear_speed: float = 1.7,
        max_angular_speed: float = 2.8,
        dt: float = 0.08,
        lidar_range: float = 5.0,
        lidar_num_rays: int = 36,
        render_mode: Optional[str] = None,
        max_steps: int = 700,
        boxes: Optional[Iterable[Any] | int] = None,
        num_boxes: Optional[int] = None,
        goal_area: Optional[Any] = None,
        goal: Optional[Sequence[float]] = None,
        spawn: Optional[Sequence[float]] = None,
        map: Any = None,
        walls: Optional[Iterable[Wall]] = None,
        pickup_distance: float = 1.25,
        pickup_height: float = 0.20,
        auto_pickup: bool = True,
        drop_height: float = 0.22,
        max_fork_height: float = 1.8,
        fork_speed: float = 1.5,
        progress_reward_scale: float = 1.0,
        action_penalty: float = 0.001,
        time_penalty: float = 0.01,
        collision_penalty: float = 0.35,
        pickup_reward: float = 1.0,
        delivery_reward: float = 8.0,
        goal_reward: float = 10.0,
        shadow_offset: Sequence[float] = (0.10, -0.10),
        scenario: Any = None,
    ) -> None:
        super().__init__()
        if world_size <= 2.0 * robot_radius + 0.2:
            raise ValueError("world_size is too small for the forklift radius")
        if int(lidar_num_rays) < 4:
            raise ValueError("lidar_num_rays must be at least 4")
        if int(max_steps) < 1:
            raise ValueError("max_steps must be positive")
        if max_fork_height <= 0.0 or fork_speed <= 0.0:
            raise ValueError("fork height and fork speed must be positive")
        for value in (progress_reward_scale, action_penalty, time_penalty,
                      collision_penalty, pickup_reward, delivery_reward,
                      goal_reward):
            if float(value) < 0.0:
                raise ValueError("reward coefficients must be non-negative")

        self.render_mode = render_mode
        if isinstance(render_mode, (list, tuple, set)):
            self._render_mode_active = next(iter(render_mode), None)
        else:
            self._render_mode_active = render_mode

        if scenario is not None and map is not None:
            raise ValueError("pass either scenario or map, not both")
        self._scenario: Optional[Scenario] = None
        if scenario is not None:
            self._scenario = load_scenario(scenario)
            map = self._scenario.map
            if boxes is None:
                boxes = self._scenario.boxes
            if goal_area is None:
                goal_area = self._scenario.goal_area

        # Keep a copy of the map input.  The map loader can return a mutable
        # object owned by a caller, so every reset rebuilds the static scene.
        self._map_input = map
        self._map_override = load_map(map) if map is not None else None
        map_dict = map if isinstance(map, dict) else {}
        map_meta = map_dict.get("meta", {}) if isinstance(map_dict, dict) else {}
        if not isinstance(map_meta, dict):
            map_meta = {}

        self.world_size = float(self._map_override.world_size) if self._map_override is not None else float(world_size)
        self.robot_radius = float(robot_radius)
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.dt = float(dt)
        self.lidar_range = float(lidar_range)
        self.lidar_num_rays = int(lidar_num_rays)
        self.max_steps = int(max_steps)
        self.pickup_distance = float(pickup_distance)
        self.pickup_height = float(pickup_height)
        self.auto_pickup = bool(auto_pickup)
        self.drop_height = float(drop_height)
        self.max_fork_height = float(max_fork_height)
        self.fork_speed = float(fork_speed)
        self.progress_reward_scale = float(progress_reward_scale)
        self.action_penalty = float(action_penalty)
        self.time_penalty = float(time_penalty)
        self.collision_penalty = float(collision_penalty)
        self.pickup_reward = float(pickup_reward)
        self.delivery_reward = float(delivery_reward)
        self.goal_reward = float(goal_reward)
        shadow_vec = np.asarray(shadow_offset, dtype=np.float32).reshape(-1)
        if shadow_vec.size != 2 or not np.all(np.isfinite(shadow_vec)):
            raise ValueError("shadow_offset must contain two finite values")
        self.shadow_offset = shadow_vec.copy()

        if goal_area is None and goal is not None:
            goal_area = goal
        if goal_area is None and isinstance(map_dict, dict) and "goal_area" in map_dict:
            goal_area = map_dict["goal_area"]

        # A map made with the existing editor may carry a labelled colliding
        # object such as ``crate``.  When no explicit box list is supplied,
        # treat those labelled objects as movable boxes and keep other object
        # elements as static shelves/obstacles.
        map_element_boxes: list[dict[str, Any]] = []
        if self._map_override is not None:
            map_label_counts: dict[str, int] = {}
            for index, element in enumerate(getattr(self._map_override, "elements", []) or []):
                kind = str(getattr(element, "kind", "")).lower()
                label = str(getattr(element, "label", "") or "")
                label_lower = label.lower()
                is_box = kind == "object" and any(word in label_lower for word in ("box", "crate", "pallet", "cargo"))
                if is_box and float(getattr(element, "w", 0.0)) > 0.0 and float(getattr(element, "h", 0.0)) > 0.0:
                    base_id = label or f"map_box_{index}"
                    suffix = map_label_counts.get(base_id, 0)
                    map_label_counts[base_id] = suffix + 1
                    box_id = base_id if suffix == 0 else f"{base_id}_{suffix}"
                    map_element_boxes.append({
                        "id": box_id,
                        "pos": [float(getattr(element, "x", 0.0)), float(getattr(element, "y", 0.0))],
                        "width": float(getattr(element, "w", 0.8)),
                        "depth": float(getattr(element, "h", 0.8)),
                        "height": 0.6,
                    })

        if self._map_override is not None:
            map_boxes_explicit = boxes is not None or (isinstance(map_dict, dict) and "boxes" in map_dict)
            self._base_walls = list(getattr(self._map_override, "walls", []) or [])
            if not map_boxes_explicit:
                # Only non-box colliding elements remain static obstacles.
                map_box_ids = {entry["id"] for entry in map_element_boxes}
                for index, element in enumerate(getattr(self._map_override, "elements", []) or []):
                    aabb = getattr(element, "aabb", None)
                    label = str(getattr(element, "label", "") or "")
                    if getattr(element, "collide", False) and aabb is not None and label not in map_box_ids:
                        self._base_walls.append(aabb)
                if boxes is None and map_element_boxes:
                    boxes = map_element_boxes
            else:
                self._base_walls = list(self._map_override.collision_walls())
            map_spawn = list(self._map_override.spawn)
            map_goal = list(self._map_override.goal)
            if spawn is None:
                spawn = map_spawn
            if goal_area is None:
                override_meta = getattr(self._map_override, "meta", {})
                if not isinstance(override_meta, dict):
                    override_meta = {}
                goal_area = map_meta.get("goal_area", override_meta.get("goal_area", map_goal))
        else:
            # Keep a few open warehouse racks in the default scene so the
            # forklift has the same CAD wall language as ``MobileRobotEnv``:
            # white AABBs with the shared ``///`` hatch.  They sit along the
            # arena edges and leave the default box/goal route unobstructed.
            if walls is None:
                scale = self.world_size / 12.0
                self._base_walls = [
                    Wall(-5.5 * scale, 2.8 * scale, -4.5 * scale, 3.1 * scale),
                    Wall(4.5 * scale, -3.5 * scale, 5.5 * scale, -3.2 * scale),
                    Wall(-5.5 * scale, -2.0 * scale, -5.0 * scale, 0.0),
                    Wall(4.8 * scale, 0.5 * scale, 5.3 * scale, 1.8 * scale),
                ]
            else:
                self._base_walls = list(walls)
        if walls is not None and self._map_override is not None:
            self._base_walls.extend(list(walls))

        if spawn is None:
            spawn = [-4.3, -4.0, 0.0]
        self._spawn_spec = self._coerce_spawn(spawn)

        # A goal may be a centre pair, [x, y, width, height],
        # {"center": ..., "size": ...}, or ([x, y], [w, h]).
        if goal_area is None:
            goal_area = [4.0, 4.0, 2.4, 2.4]
        self._goal_center_spec, self._goal_size_spec = self._coerce_goal_area(goal_area)

        if boxes is None and isinstance(map_dict, dict) and "boxes" in map_dict:
            boxes = map_dict["boxes"]
        if boxes is None and "boxes" in map_meta:
            boxes = map_meta["boxes"]
        if boxes is None and self._map_override is not None:
            override_meta = getattr(self._map_override, "meta", {})
            if isinstance(override_meta, dict) and "boxes" in override_meta:
                boxes = override_meta["boxes"]
        if isinstance(boxes, (int, np.integer)):
            num_boxes = int(boxes)
            boxes = None
        self._num_boxes = None if num_boxes is None else int(num_boxes)
        if self._num_boxes is not None and self._num_boxes < 0:
            raise ValueError("num_boxes must be non-negative")
        self._box_specs = None if boxes is None else self._normalize_box_specs(boxes)
        if self._box_specs is not None and self._num_boxes is not None:
            raise ValueError("pass either boxes or num_boxes, not both")

        # Public mutable state is initialized here as well as in reset so a
        # caller can inspect or render an environment before its first reset.
        self.walls: list[Wall] = []
        self.boxes: list[ForkliftBox] = []
        self.goal_area_center = np.asarray(self._goal_center_spec, dtype=np.float32).copy()
        self.goal_area_size = np.asarray(self._goal_size_spec, dtype=np.float32).copy()
        self._fit_goal_area_to_world()
        self.pos = np.asarray(self._spawn_spec[:2], dtype=np.float32).copy()
        self.theta = float(self._spawn_spec[2])
        self.vel = np.zeros(2, dtype=np.float32)
        self.fork_height = 0.0
        self.carried_box_id: Optional[str] = None
        self.target_box_id: Optional[str] = None
        self.goal = self.goal_area_center.copy()
        self._goal_kind = "complete"
        self._steps = 0
        self._prev_dist = 0.0
        self._last_reward = 0.0
        self._episode_reward = 0.0
        self._last_collision = False
        # Last normalized control [u, v, lift].  The first two channels are
        # shown as compact behaviour bars in the BEV renderer.
        self._last_action = np.zeros(3, dtype=np.float32)
        self._last_fig = None
        self._trail: list[list[float]] = []
        # Optional scripted traffic used by ForkliftParallelEnv.  The base
        # forklift remains single-agent when this registry is empty.
        self.extras: dict[str, AgentRecord] = {}

        # Lidar plus robot-frame goal, velocity, fork and task-state features.
        # The first eight non-lidar entries are:
        # goal-forward, goal-left, linear speed, angular speed, fork height,
        # carrying mask, remaining fraction, and goal kind (0/1/2).
        extra_low = np.array([
            -2.0 * self.world_size, -2.0 * self.world_size,
            -self.max_linear_speed, -self.max_angular_speed,
            0.0, 0.0, 0.0, 0.0,
        ], dtype=np.float32)
        extra_high = np.array([
            2.0 * self.world_size, 2.0 * self.world_size,
            self.max_linear_speed, self.max_angular_speed,
            self.max_fork_height, 1.0, 1.0, 2.0,
        ], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([np.zeros(self.lidar_num_rays, dtype=np.float32), extra_low]),
            high=np.concatenate([np.full(self.lidar_num_rays, self.lidar_range, dtype=np.float32), extra_high]),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-np.ones(3, dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Configuration and reset helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_spawn(value: Sequence[float]) -> tuple[float, float, float]:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size < 2:
            raise ValueError("spawn must contain x and y")
        return (float(arr[0]), float(arr[1]), float(arr[2]) if arr.size > 2 else 0.0)

    @staticmethod
    def _coerce_goal_area(value: Any) -> tuple[tuple[float, float], tuple[float, float]]:
        if isinstance(value, dict):
            centre = value.get("center", value.get("centre", value.get("pos", value.get("goal", [0.0, 0.0]))))
            size = value.get("size", [value.get("width", 2.4), value.get("height", 2.4)])
            c = np.asarray(centre, dtype=np.float32).reshape(-1)
            s = np.asarray(size, dtype=np.float32).reshape(-1)
        elif isinstance(value, (tuple, list)) and len(value) == 2 and np.asarray(value[0]).size == 2 and np.asarray(value[1]).size == 2:
            c = np.asarray(value[0], dtype=np.float32).reshape(-1)
            s = np.asarray(value[1], dtype=np.float32).reshape(-1)
        else:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size < 2:
                raise ValueError("goal_area needs a centre")
            c = arr[:2]
            s = arr[2:4] if arr.size >= 4 else np.array([2.4, 2.4], dtype=np.float32)
        if c.size < 2 or s.size < 2:
            raise ValueError("goal_area centre and size need two components")
        if float(s[0]) <= 0.0 or float(s[1]) <= 0.0:
            raise ValueError("goal_area width and height must be positive")
        return (float(c[0]), float(c[1])), (float(s[0]), float(s[1]))

    def _fit_goal_area_to_world(self) -> None:
        """Keep the delivery rectangle reachable in small custom arenas."""
        half = self.world_size / 2.0
        max_size = max(0.2, self.world_size - 0.1)
        self.goal_area_size[:] = np.minimum(self.goal_area_size, max_size)
        for axis in (0, 1):
            limit = max(0.0, half - float(self.goal_area_size[axis]) / 2.0)
            self.goal_area_center[axis] = float(np.clip(self.goal_area_center[axis], -limit, limit))

    @staticmethod
    def _spec_from_value(value: Any, index: int) -> ForkliftBox:
        if isinstance(value, ForkliftBox):
            box = value.copy()
            box.id = str(box.id or f"box_{index}")
            return box
        if isinstance(value, dict):
            ident = value.get("id", value.get("name", f"box_{index}"))
            pos = value.get("pos", value.get("position", value.get("center", value.get("centre", [value.get("x", 0.0), value.get("y", 0.0)]))))
            size = value.get("size", [value.get("width", value.get("w", 0.8)),
                                       value.get("depth", value.get("d", value.get("height_xy", 0.8)))])
            size = np.asarray(size, dtype=np.float32).reshape(-1)
            return ForkliftBox(
                id=str(ident), pos=np.asarray(pos, dtype=np.float32)[:2],
                width=float(size[0]),
                depth=float(size[1] if size.size > 1 else size[0]),
                height=float(value.get("height", value.get("z", 0.6))),
                stack_level=int(value.get("stack_level", value.get("level", 0))),
                theta=float(value.get("theta", value.get("angle", value.get("orientation", 0.0)))),
            )
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size < 2:
            raise ValueError("each box needs an x and y position")
        # A third tuple value is useful for explicitly describing a source
        # stack level; dimensions stay at their documented defaults.
        if arr.size >= 4:
            return ForkliftBox(
                id=f"box_{index}", pos=arr[:2], width=float(arr[2]), depth=float(arr[3]),
                height=float(arr[4]) if arr.size > 4 else 0.6,
                stack_level=int(arr[5]) if arr.size > 5 else 0,
                theta=float(arr[6]) if arr.size > 6 else 0.0,
            )
        level = int(arr[2]) if arr.size == 3 and float(arr[2]).is_integer() and arr[2] >= 0 else 0
        return ForkliftBox(id=f"box_{index}", pos=arr[:2], stack_level=level)

    def _normalize_box_specs(self, values: Iterable[Any]) -> list[ForkliftBox]:
        # Accept a single [x, y] pair as one box instead of treating x and y
        # as two independent records.
        if isinstance(values, ForkliftBox) or isinstance(values, dict):
            values = [values]
        elif isinstance(values, np.ndarray) and values.ndim == 1 and values.size >= 2:
            values = [values]
        elif isinstance(values, (list, tuple)) and len(values) >= 2 and all(np.isscalar(v) for v in values):
            values = [values]
        else:
            values = list(values)
        expanded: list[Any] = []
        for value in values:
            if isinstance(value, dict):
                # ``count``/``stack_count`` are convenient for JSON maps.  A
                # numeric ``stack`` is accepted as the same shorthand when no
                # explicit level was supplied.  Explicit ``stack_level`` and
                # ``level`` still describe one box at a chosen level.
                count_value = value.get("count", value.get("stack_count", value.get("stack_height", value.get("stacked"))))
                if count_value is None and "stack" in value and "stack_level" not in value and "level" not in value:
                    candidate = value.get("stack")
                    if isinstance(candidate, (int, np.integer)) and int(candidate) > 1:
                        count_value = candidate
                count = int(count_value) if count_value is not None else 1
                if count < 1:
                    raise ValueError("a box stack count must be at least one")
                if count == 1:
                    expanded.append(value)
                else:
                    for level in range(count):
                        item = dict(value)
                        item.pop("count", None)
                        item.pop("stack_count", None)
                        item.pop("stack", None)
                        item["stack_level"] = level
                        base_id = str(item.get("id", item.get("name", f"box_{len(expanded)}")))
                        item["id"] = f"{base_id}_{level}"
                        expanded.append(item)
            else:
                expanded.append(value)
        specs = [self._spec_from_value(v, i) for i, v in enumerate(expanded)]
        if len({s.id for s in specs}) != len(specs):
            raise ValueError("box ids must be unique")
        self._assign_stack_levels(specs)
        return specs

    @staticmethod
    def _same_xy(a: Sequence[float], b: Sequence[float], tol: float = 1e-4) -> bool:
        return float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2])) <= tol

    def _assign_stack_levels(self, boxes: list[ForkliftBox]) -> None:
        # Group boxes with the same source centre.  Explicit levels are kept
        # only when they form a unique ordering; all other inputs use list
        # order, which makes repeated positions a convenient stack notation.
        groups: list[list[ForkliftBox]] = []
        for box in boxes:
            group = next((g for g in groups if self._same_xy(g[0].source_pos, box.source_pos)), None)
            if group is None:
                groups.append([box])
            else:
                group.append(box)
        for group in groups:
            group.sort(key=lambda b: b.stack_level)
            stack_z = 0.0
            for level, box in enumerate(group):
                box.stack_level = level
                box.source_pos = box.pos.copy()
                box.z = stack_z
                stack_z += box.height

    def _default_box_specs(self) -> list[ForkliftBox]:
        # Two boxes at the north-west pallet deliberately exercise stacking.
        return self._normalize_box_specs([
            {"id": "box_0", "pos": [-2.5, 2.0]},
            {"id": "box_1", "pos": [-2.5, 2.0]},
            {"id": "box_2", "pos": [0.8, -1.2]},
            {"id": "box_3", "pos": [2.8, 1.7]},
        ])

    def _generate_box_specs(self, rng: np.random.Generator, count: int) -> list[ForkliftBox]:
        if count <= 0:
            return []
        half = self.world_size / 2.0 - self.robot_radius - 0.65
        goal = self.goal_area_center
        occupied: list[np.ndarray] = [self.pos.copy(), goal.copy()]
        out: list[ForkliftBox] = []
        for i in range(count):
            # About one quarter of generated boxes form a two-box pallet when
            # the requested count allows it.
            for _ in range(120):
                p = np.array([rng.uniform(-half, half), rng.uniform(-half, half)], dtype=np.float32)
                if any(float(np.linalg.norm(p - q)) < 1.35 for q in occupied):
                    continue
                if self._point_in_wall(p, margin=0.5):
                    continue
                occupied.append(p)
                out.append(ForkliftBox(id=f"box_{i}", pos=p))
                break
            else:
                # A deterministic fallback remains valid in very small arenas.
                p = np.array([0.0, 0.0], dtype=np.float32)
                out.append(ForkliftBox(id=f"box_{i}", pos=p))
        if count >= 2:
            # Put the last two boxes together, while retaining deterministic
            # ids and all caller-requested box dimensions.
            out[-1].pos = out[-2].pos.copy()
            out[-1].source_pos = out[-2].pos.copy()
        self._assign_stack_levels(out)
        return out

    def _clone_boxes(self, specs: Optional[Iterable[Any]], rng: np.random.Generator) -> list[ForkliftBox]:
        if specs is None:
            if self._num_boxes is not None:
                return self._generate_box_specs(rng, self._num_boxes)
            specs = self._default_box_specs()
        elif not isinstance(specs, list):
            specs = self._normalize_box_specs(specs)
        return [self._spec_from_value(b, i) for i, b in enumerate(specs)]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        rng = self.np_random
        options = options or {}

        if self._map_override is not None:
            # ``_base_walls`` may intentionally exclude labelled movable map
            # objects; copy the primitives so later map mutations are isolated.
            self.walls = [Wall(w.x_lo, w.y_lo, w.x_hi, w.y_hi) for w in self._base_walls]
        else:
            self.walls = [Wall(w.x_lo, w.y_lo, w.x_hi, w.y_hi) for w in self._base_walls]

        spawn_spec = options.get("spawn", self._spawn_spec)
        sx, sy, st = self._coerce_spawn(spawn_spec)
        self.pos = np.array([sx, sy], dtype=np.float32)
        self.theta = float(st)
        self.vel = np.zeros(2, dtype=np.float32)
        self.fork_height = float(options.get("fork_height", 0.0))
        self.fork_height = float(np.clip(self.fork_height, 0.0, self.max_fork_height))
        self.carried_box_id = None

        goal_value = options.get("goal_area")
        if goal_value is not None:
            centre, size = self._coerce_goal_area(goal_value)
            self.goal_area_center = np.asarray(centre, dtype=np.float32)
            self.goal_area_size = np.asarray(size, dtype=np.float32)
        else:
            self.goal_area_center = np.asarray(self._goal_center_spec, dtype=np.float32).copy()
            self.goal_area_size = np.asarray(self._goal_size_spec, dtype=np.float32).copy()
        self._fit_goal_area_to_world()

        input_boxes = options.get("boxes", None)
        if input_boxes is None:
            self.boxes = self._clone_boxes(self._box_specs, rng)
        else:
            self.boxes = self._normalize_box_specs(input_boxes)
            self.boxes = [b.copy() for b in self.boxes]
        self._assign_stack_levels(self.boxes)
        for box in self.boxes:
            box.picked = False
            box.carried = False
            box.delivered = False
            box.goal_level = -1
            box.pos = box.source_pos.copy()
            self._nudge_box_clear(box)

        self._ensure_clear(self.pos, self.robot_radius)
        # Randomized maps sample a free chassis point first.  If the sampled
        # heading puts a fork tine through a shelf, rotate to the nearest clear
        # heading so every episode starts from a physically valid pose.
        if not self._forks_clear_of_walls(self.pos, self.theta):
            original_theta = float(self.theta)
            offsets = np.linspace(
                math.pi / 16.0, 2.0 * math.pi - math.pi / 16.0, 32,
                dtype=np.float32,
            )
            for offset in offsets:
                candidate_theta = original_theta + float(offset)
                if self._forks_clear_of_walls(self.pos, candidate_theta):
                    self.theta = (candidate_theta + math.pi) % (2.0 * math.pi) - math.pi
                    break
        self._steps = 0
        self._trail = [self.pos.tolist()]
        self._last_reward = 0.0
        self._episode_reward = 0.0
        self._last_collision = False
        self._last_action[:] = 0.0
        self._last_fig = None
        for rec in self.extras.values():
            rec.vel[:] = 0.0
            rec.last_action[:] = 0.0
            rec.last_collided = False
            rec.stuck_steps = 0
        self._update_goal()
        self._prev_dist = self._distance_to_goal()
        obs = self._get_obs()
        return obs, self._info()

    # ------------------------------------------------------------------
    # Task state and goal selection
    # ------------------------------------------------------------------
    def _available_boxes(self) -> list[ForkliftBox]:
        result = []
        for box in self.boxes:
            if box.delivered or box.carried:
                continue
            # Only the top box at a source pallet can be picked.
            blocked = any(
                (not other.delivered and not other.carried and other is not box
                 and self._same_xy(other.source_pos, box.source_pos)
                 and other.stack_level > box.stack_level)
                for other in self.boxes
            )
            if not blocked:
                result.append(box)
        return result

    def _find_box(self, box_id: Optional[str]) -> Optional[ForkliftBox]:
        if box_id is None:
            return None
        return next((b for b in self.boxes if b.id == str(box_id)), None)

    def get_box(self, box_id: str) -> ForkliftBox:
        box = self._find_box(box_id)
        if box is None:
            raise KeyError(f"unknown box {box_id!r}")
        return box

    @property
    def carried_box(self) -> Optional[ForkliftBox]:
        return self._find_box(self.carried_box_id)

    @property
    def current_goal(self) -> np.ndarray:
        """Alias for the active dynamic goal (a box or the delivery area)."""
        return self.goal

    @property
    def goal_area(self) -> dict[str, np.ndarray]:
        return {
            "center": self.goal_area_center.copy(),
            "size": self.goal_area_size.copy(),
        }

    @property
    def goal_type(self) -> str:
        return self._goal_kind

    @property
    def goal_kind(self) -> str:
        return self._goal_kind

    @property
    def holding_box(self) -> bool:
        return self.carried_box_id is not None

    @property
    def delivered_count(self) -> int:
        return sum(1 for b in self.boxes if b.delivered)

    @property
    def remaining_count(self) -> int:
        return sum(1 for b in self.boxes if not b.delivered)

    @property
    def all_boxes_delivered(self) -> bool:
        # An empty task is already complete; this is useful for curriculum
        # stages that deliberately start with zero boxes.
        return self.remaining_count == 0

    # ------------------------------------------------------------------
    # Optional scripted traffic agents
    # ------------------------------------------------------------------
    def list_agents(self) -> list[str]:
        """Return the ids of background agents currently in the scene."""
        return list(self.extras)

    def get_agent(self, agent_id: str) -> AgentRecord:
        """Return a mutable background-agent record for a scripted policy."""
        try:
            return self.extras[str(agent_id)]
        except KeyError as exc:
            raise KeyError(f"unknown forklift agent {agent_id!r}") from exc

    def add_agent(self, agent_id: str, *, pos: Optional[Sequence[float]] = None,
                  theta: float = 0.0, goal: Optional[Sequence[float]] = None,
                  method: Optional[Callable] = None,
                  controllable: bool = True, color: Optional[tuple] = None,
                  radius: Optional[float] = None) -> AgentRecord:
        """Register a moving background robot for a parallel rollout.

        The agent is not added to the forklift action space.  Its policy is
        called internally once per simulator step, matching the mobile-robot
        traffic adapter.
        """
        ident = str(agent_id)
        if ident == "main":
            raise ValueError("the 'main' id is reserved for the forklift")
        if ident in self.extras:
            raise ValueError(f"agent id already exists: {ident!r}")
        rec_radius = self.robot_radius if radius is None else float(radius)
        if rec_radius <= 0.0:
            raise ValueError("agent radius must be positive")
        point = (self.pos.copy() if pos is None
                 else np.asarray(pos, dtype=np.float32).reshape(-1)[:2].copy())
        if point.size < 2:
            raise ValueError("agent pos must contain x and y")
        self._ensure_clear(point, rec_radius)
        # Scenario generators keep traffic clear of the chassis centre, while
        # the physical forks reach farther forward.  Nudge a requested start
        # to the nearest free point when it would begin inside a tine, pallet,
        # or another registered agent.
        if not self._agent_position_valid(point, rec_radius):
            origin = point.copy()
            candidate_point: Optional[np.ndarray] = None
            for distance in np.linspace(0.25, 2.0, 8):
                for angle in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False):
                    candidate = origin + np.asarray([
                        math.cos(float(angle)) * float(distance),
                        math.sin(float(angle)) * float(distance),
                    ], dtype=np.float32)
                    self._ensure_clear(candidate, rec_radius)
                    if self._agent_position_valid(candidate, rec_radius):
                        candidate_point = candidate
                        break
                if candidate_point is not None:
                    break
            if candidate_point is not None:
                point = candidate_point
        target = (self.goal.copy() if goal is None
                  else np.asarray(goal, dtype=np.float32).reshape(-1)[:2].copy())
        if target.size < 2:
            raise ValueError("agent goal must contain x and y")
        self.extras[ident] = AgentRecord(
            id=ident,
            pos=point,
            theta=float(theta),
            vel=np.zeros(2, dtype=np.float32),
            goal=target,
            method=method,
            controllable=bool(controllable),
            color=color,
            radius=rec_radius,
        )
        return self.extras[ident]

    def clear_agents(self) -> None:
        """Remove all background traffic agents."""
        self.extras.clear()

    def _agent_obs(self, rec: AgentRecord) -> np.ndarray:
        """Compact goal-relative observation passed to traffic policies."""
        delta = rec.goal - rec.pos
        return np.asarray([
            float(delta[0]), float(delta[1]),
            float(rec.vel[0]), float(rec.vel[1]),
        ], dtype=np.float32)

    def set_pos(self, pos: Sequence[float], *, theta: Optional[float] = None,
                reset_velocities: bool = True) -> np.ndarray:
        """Teleport the forklift to an open point for scripted demonstrations."""
        point = np.asarray(pos, dtype=np.float32).reshape(-1)
        if point.size < 2:
            raise ValueError("pos must contain x and y")
        candidate = point[:2].copy()
        self._ensure_clear(candidate, self.robot_radius)
        if theta is not None:
            self.theta = (float(theta) + math.pi) % (2.0 * math.pi) - math.pi
        if self.carried_box is not None:
            self.pos = candidate
            self.carried_box.pos = self._carried_position()
            self.carried_box.theta = self.theta
        else:
            self.pos = candidate
        if reset_velocities:
            self.vel[:] = 0.0
        self._update_goal()
        self._prev_dist = self._distance_to_goal()
        return self.pos

    def set_box_pos(self, box_id: str, pos: Sequence[float]) -> np.ndarray:
        """Move an uncarried source box for scenario setup."""
        box = self.get_box(box_id)
        if box.carried or box.delivered:
            raise ValueError("only an available source box can be repositioned")
        point = np.asarray(pos, dtype=np.float32).reshape(-1)
        if point.size < 2:
            raise ValueError("box position must contain x and y")
        box.pos = point[:2].copy()
        box.source_pos = box.pos.copy()
        self._nudge_box_clear(box)
        self._assign_stack_levels(self.boxes)
        self._update_goal()
        self._prev_dist = self._distance_to_goal()
        return box.pos

    def _update_goal(self) -> None:
        carried = self.carried_box
        if carried is not None:
            self.goal = self.goal_area_center.copy()
            self.target_box_id = carried.id
            self._goal_kind = "delivery"
            return
        available = self._available_boxes()
        if available:
            available.sort(key=lambda b: (float(np.linalg.norm(b.pos - self.pos)), b.id))
            target = available[0]
            self.target_box_id = target.id
            self.goal = target.pos.copy()
            self._goal_kind = "box"
        else:
            self.target_box_id = None
            self.goal = self.goal_area_center.copy()
            self._goal_kind = "complete"

    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.goal - self.pos))

    def _goal_label(self) -> str:
        if self._goal_kind == "box":
            return f"box:{self.target_box_id}"
        if self._goal_kind == "delivery":
            return "goal_area"
        return "complete"

    # ------------------------------------------------------------------
    # Geometry and collision
    # ------------------------------------------------------------------
    def _point_in_wall(self, point: Sequence[float], margin: float = 0.0) -> bool:
        x, y = float(point[0]), float(point[1])
        for wall in self.walls:
            if wall.x_lo - margin <= x <= wall.x_hi + margin and wall.y_lo - margin <= y <= wall.y_hi + margin:
                return True
        return False

    def _nudge_box_clear(self, box: ForkliftBox) -> None:
        half = self.world_size / 2.0
        extent_x, extent_y = self._obb_extents(box.width, box.depth, box.theta)
        margin_x = extent_x + 0.02
        margin_y = extent_y + 0.02
        box.pos[0] = float(np.clip(box.pos[0], -half + margin_x, half - margin_x))
        box.pos[1] = float(np.clip(box.pos[1], -half + margin_y, half - margin_y))
        # Boxes are allowed to share a source position (a stack), but not start
        # embedded in a static wall.  Push a malformed input to the nearest
        # free side rather than failing an otherwise useful episode.
        for _ in range(20):
            hit = False
            for wall in self.walls:
                if self._obb_overlaps_aabb(
                    box.pos, box.width, box.depth, box.theta, wall, margin=0.0,
                ):
                    hit = True
                    left = abs(box.pos[0] - (wall.x_lo - extent_x))
                    right = abs((wall.x_hi + extent_x) - box.pos[0])
                    down = abs(box.pos[1] - (wall.y_lo - extent_y))
                    up = abs((wall.y_hi + extent_y) - box.pos[1])
                    side = min((left, -1), (right, 1), (down, -2), (up, 2), key=lambda p: p[0])[1]
                    if side == -1:
                        box.pos[0] = wall.x_lo - extent_x - 0.03
                    elif side == 1:
                        box.pos[0] = wall.x_hi + extent_x + 0.03
                    elif side == -2:
                        box.pos[1] = wall.y_lo - extent_y - 0.03
                    else:
                        box.pos[1] = wall.y_hi + extent_y + 0.03
            if not hit:
                break
            box.pos[0] = float(np.clip(box.pos[0], -half + margin_x, half - margin_x))
            box.pos[1] = float(np.clip(box.pos[1], -half + margin_y, half - margin_y))
        box.source_pos = box.pos.copy()

    @staticmethod
    def _obb_axes(theta: float) -> tuple[np.ndarray, np.ndarray]:
        c, s = math.cos(float(theta)), math.sin(float(theta))
        return np.array([c, s], dtype=np.float32), np.array([-s, c], dtype=np.float32)

    @classmethod
    def _obb_corners(cls, pos: Sequence[float], width: float, depth: float,
                     theta: float) -> np.ndarray:
        centre = np.asarray(pos, dtype=np.float32).reshape(2)
        axis_w, axis_d = cls._obb_axes(theta)
        half_w, half_d = float(width) / 2.0, float(depth) / 2.0
        return np.asarray([
            centre - axis_w * half_w - axis_d * half_d,
            centre + axis_w * half_w - axis_d * half_d,
            centre + axis_w * half_w + axis_d * half_d,
            centre - axis_w * half_w + axis_d * half_d,
        ], dtype=np.float32)

    @staticmethod
    def _obb_extents(width: float, depth: float, theta: float) -> tuple[float, float]:
        c, s = abs(math.cos(float(theta))), abs(math.sin(float(theta)))
        return c * float(width) / 2.0 + s * float(depth) / 2.0, \
            s * float(width) / 2.0 + c * float(depth) / 2.0

    def _fork_obbs(
        self,
        pos: Optional[Sequence[float]] = None,
        theta: Optional[float] = None,
    ) -> list[tuple[np.ndarray, float, float, float]]:
        """Return the two physical fork tines as thin oriented rectangles.

        The chassis uses a circular collision footprint, but the forks extend
        well beyond it.  Keeping the tines as OBBs makes a low fork catch a
        shelf or traffic robot before the chassis reaches it while preserving
        the separate pallet pickup geometry.
        """
        point = np.asarray(self.pos if pos is None else pos, dtype=np.float32).reshape(2)
        heading = self.theta if theta is None else float(theta)
        c, s = math.cos(heading), math.sin(heading)
        forward = np.asarray([c, s], dtype=np.float32)
        lateral = np.asarray([-s, c], dtype=np.float32)
        body_width = self.robot_radius * 1.55
        fork_base = self.robot_radius * 0.8
        fork_tip = self.robot_radius + 1.15
        fork_length = max(0.10, fork_tip - fork_base)
        fork_width = max(0.10, self.robot_radius * 0.16)
        centre_forward = fork_base + fork_length / 2.0
        side_offset = body_width * 0.33
        return [
            (
                point + forward * centre_forward + lateral * (side * side_offset),
                fork_length,
                fork_width,
                heading,
            )
            for side in (-1.0, 1.0)
        ]

    def _forks_clear_of_walls(self, pos: Sequence[float], theta: float) -> bool:
        """Check only static walls, used to make generated starts valid."""
        for centre, length, width, fork_theta in self._fork_obbs(pos, theta):
            for wall in self.walls:
                if self._obb_overlaps_aabb(
                    centre, length, width, fork_theta, wall, margin=0.01,
                ):
                    return False
        return True

    @staticmethod
    def _project_interval(points: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
        values = np.asarray(points, dtype=np.float32) @ np.asarray(axis, dtype=np.float32)
        return float(np.min(values)), float(np.max(values))

    @classmethod
    def _obb_overlaps_aabb(cls, pos: Sequence[float], width: float, depth: float,
                           theta: float, wall: Wall, margin: float = 0.0) -> bool:
        obb = cls._obb_corners(pos, width, depth, theta)
        aabb = np.asarray([
            [wall.x_lo, wall.y_lo], [wall.x_hi, wall.y_lo],
            [wall.x_hi, wall.y_hi], [wall.x_lo, wall.y_hi],
        ], dtype=np.float32)
        axis_w, axis_d = cls._obb_axes(theta)
        axes = (axis_w, axis_d, np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32))
        for axis in axes:
            alo, ahi = cls._project_interval(obb, axis)
            blo, bhi = cls._project_interval(aabb, axis)
            if ahi < blo - margin or bhi < alo - margin:
                return False
        return True

    @classmethod
    def _obb_overlaps_obb(cls, pos_a: Sequence[float], width_a: float, depth_a: float,
                          theta_a: float, pos_b: Sequence[float], width_b: float,
                          depth_b: float, theta_b: float, margin: float = 0.0) -> bool:
        a = cls._obb_corners(pos_a, width_a, depth_a, theta_a)
        b = cls._obb_corners(pos_b, width_b, depth_b, theta_b)
        a_w, a_d = cls._obb_axes(theta_a)
        b_w, b_d = cls._obb_axes(theta_b)
        for axis in (a_w, a_d, b_w, b_d):
            alo, ahi = cls._project_interval(a, axis)
            blo, bhi = cls._project_interval(b, axis)
            if ahi < blo - margin or bhi < alo - margin:
                return False
        return True

    @classmethod
    def _circle_overlaps_obb(cls, point: Sequence[float], radius: float,
                             box_pos: Sequence[float], width: float, depth: float,
                             theta: float) -> bool:
        centre = np.asarray(box_pos, dtype=np.float32).reshape(2)
        rel = np.asarray(point, dtype=np.float32).reshape(2) - centre
        axis_w, axis_d = cls._obb_axes(theta)
        local = np.array([float(np.dot(rel, axis_w)), float(np.dot(rel, axis_d))])
        closest = np.clip(local, [-float(width) / 2.0, -float(depth) / 2.0],
                          [float(width) / 2.0, float(depth) / 2.0])
        delta = local - closest
        return float(np.dot(delta, delta)) < float(radius) ** 2

    @classmethod
    def _ray_obb_t(cls, px: float, py: float, dx: float, dy: float,
                   box: ForkliftBox, inflation: float, max_range: float) -> float:
        """Return the first ray hit with an oriented box, or max_range."""
        rel = np.asarray([px, py], dtype=np.float32) - box.pos
        axis_w, axis_d = cls._obb_axes(box.theta)
        origin = np.asarray([float(np.dot(rel, axis_w)), float(np.dot(rel, axis_d))])
        direction = np.asarray([float(np.dot([dx, dy], axis_w)),
                                float(np.dot([dx, dy], axis_d))])
        half_w = float(box.width) / 2.0 + float(inflation)
        half_d = float(box.depth) / 2.0 + float(inflation)
        t_min, t_max = 0.0, float(max_range)
        for value, velocity, extent in ((origin[0], direction[0], half_w),
                                        (origin[1], direction[1], half_d)):
            if abs(float(velocity)) < 1e-9:
                if abs(float(value)) > extent:
                    return float(max_range)
                continue
            t1 = (-extent - float(value)) / float(velocity)
            t2 = (extent - float(value)) / float(velocity)
            if t1 > t2:
                t1, t2 = t2, t1
            t_min, t_max = max(t_min, t1), min(t_max, t2)
            if t_min > t_max:
                return float(max_range)
        return float(np.clip(t_min, 0.0, max_range))

    @staticmethod
    def _aabb_overlaps_wall(pos: Sequence[float], width: float, depth: float, wall: Wall, margin: float = 0.0) -> bool:
        return not (
            float(pos[0]) + width / 2.0 + margin <= wall.x_lo
            or float(pos[0]) - width / 2.0 - margin >= wall.x_hi
            or float(pos[1]) + depth / 2.0 + margin <= wall.y_lo
            or float(pos[1]) - depth / 2.0 - margin >= wall.y_hi
        )

    @staticmethod
    def _circle_overlaps_aabb(pos: Sequence[float], radius: float, wall: Wall) -> bool:
        cx = min(max(float(pos[0]), wall.x_lo), wall.x_hi)
        cy = min(max(float(pos[1]), wall.y_lo), wall.y_hi)
        dx = float(pos[0]) - cx
        dy = float(pos[1]) - cy
        return dx * dx + dy * dy < float(radius) ** 2

    @staticmethod
    def _circles_overlap(pos_a: Sequence[float], radius_a: float,
                         pos_b: Sequence[float], radius_b: float) -> bool:
        delta = np.asarray(pos_a, dtype=np.float32) - np.asarray(pos_b, dtype=np.float32)
        limit = float(radius_a) + float(radius_b)
        return float(np.dot(delta, delta)) < limit * limit

    @staticmethod
    def _ray_circle_t(px: float, py: float, dx: float, dy: float,
                      centre: Sequence[float], radius: float,
                      max_range: float) -> float:
        """Return the first ray hit with an inflated circular traffic robot."""
        rel_x = float(px) - float(centre[0])
        rel_y = float(py) - float(centre[1])
        projection = rel_x * float(dx) + rel_y * float(dy)
        discriminant = projection * projection - (rel_x * rel_x + rel_y * rel_y - float(radius) ** 2)
        if discriminant < 0.0:
            return float(max_range)
        root = math.sqrt(max(0.0, discriminant))
        hit = -projection - root
        if hit < 0.0:
            hit = -projection + root
        return float(np.clip(hit, 0.0, max_range)) if hit >= 0.0 else float(max_range)

    def _ensure_clear(self, point: np.ndarray, radius: Optional[float] = None) -> None:
        r = self.robot_radius if radius is None else float(radius)
        half = self.world_size / 2.0
        point[0] = float(np.clip(point[0], -half + r, half - r))
        point[1] = float(np.clip(point[1], -half + r, half - r))
        for _ in range(20):
            changed = False
            for wall in self.walls:
                if self._circle_overlaps_aabb(point, r, wall):
                    cx = min(max(float(point[0]), wall.x_lo), wall.x_hi)
                    cy = min(max(float(point[1]), wall.y_lo), wall.y_hi)
                    dx, dy = float(point[0] - cx), float(point[1] - cy)
                    if abs(dx) + abs(dy) > 1e-8:
                        norm = math.hypot(dx, dy)
                        point[0] += (r - norm + 0.02) * dx / norm
                        point[1] += (r - norm + 0.02) * dy / norm
                    else:
                        distances = [(abs(point[0] - wall.x_lo), -1, 0),
                                     (abs(wall.x_hi - point[0]), 1, 0),
                                     (abs(point[1] - wall.y_lo), 0, -1),
                                     (abs(wall.y_hi - point[1]), 0, 1)]
                        _, sx, sy = min(distances, key=lambda v: v[0])
                        point[0] += sx * (r + 0.02)
                        point[1] += sy * (r + 0.02)
                    point[0] = float(np.clip(point[0], -half + r, half - r))
                    point[1] = float(np.clip(point[1], -half + r, half - r))
                    changed = True
            if not changed:
                break

    def _box_wall(self, box: ForkliftBox, pos: Optional[Sequence[float]] = None) -> Wall:
        """Return the AABB envelope of a pallet's oriented footprint."""
        p = box.pos if pos is None else pos
        corners = self._obb_corners(p, box.width, box.depth, box.theta)
        return Wall(float(np.min(corners[:, 0])), float(np.min(corners[:, 1])),
                    float(np.max(corners[:, 0])), float(np.max(corners[:, 1])))

    def _agent_position_valid(self, pos: Sequence[float], radius: float,
                              current: Optional[AgentRecord] = None) -> bool:
        """Check a traffic robot against walls, pallets, and live robots."""
        half = self.world_size / 2.0
        if abs(float(pos[0])) > half - radius or abs(float(pos[1])) > half - radius:
            return False
        for wall in self.walls:
            if self._circle_overlaps_aabb(pos, radius, wall):
                return False
        for box in self._obstacle_boxes():
            if self._circle_overlaps_obb(pos, radius + 0.02,
                                          box.pos, box.width, box.depth, box.theta):
                return False
        if self._circles_overlap(pos, radius, self.pos, self.robot_radius):
            return False
        for fork_pos, fork_length, fork_width, fork_theta in self._fork_obbs():
            if self._circle_overlaps_obb(
                pos, radius + 0.02, fork_pos, fork_length, fork_width, fork_theta,
            ):
                return False
        carried = self.carried_box
        if carried is not None and self._circle_overlaps_obb(
            pos, radius + 0.02, self._carried_position(),
            carried.width, carried.depth, carried.theta,
        ):
            return False
        for other in self.extras.values():
            if other is current:
                continue
            other_radius = other.radius or self.robot_radius
            if self._circles_overlap(pos, radius, other.pos, other_radius):
                return False
        return True

    def _step_extra_agents(self) -> None:
        """Advance scripted traffic once after the main forklift action."""
        if not self.extras:
            return
        for rec in list(self.extras.values()):
            # Keep the previous contact visible while the policy chooses its
            # next waypoint.  Clearing it before calling ``goal_seek_policy``
            # made a robot that reached a shelf keep steering into that shelf
            # forever, which looked like traffic stopping after a while.
            if rec.method is None:
                action = rec.default_action
            else:
                action = rec.method(self._agent_obs(rec), self)
            rec.last_collided = False
            values = np.asarray(action, dtype=np.float32).reshape(-1)
            if values.size < 2:
                values = np.pad(values, (0, 2 - values.size))
            values = np.clip(values[:2], -1.0, 1.0).astype(np.float32)
            rec.last_action[:] = values
            radius = rec.radius or self.robot_radius
            rec.vel[0] = float(values[0]) * self.max_linear_speed
            rec.vel[1] = float(values[1]) * self.max_angular_speed
            new_theta = float(rec.theta + rec.vel[1] * self.dt)
            new_pos = rec.pos + np.asarray([
                math.cos(rec.theta) * rec.vel[0] * self.dt,
                math.sin(rec.theta) * rec.vel[0] * self.dt,
            ], dtype=np.float32)
            if not self._agent_position_valid(new_pos, radius, rec):
                rec.last_collided = True
                rec.vel[0] = 0.0
                new_pos = rec.pos.copy()
            rec.pos[:] = np.asarray(new_pos, dtype=np.float32)
            rec.theta = (new_theta + math.pi) % (2.0 * math.pi) - math.pi

    def _position_valid(self, pos: Sequence[float], carried: Optional[ForkliftBox] = None,
                        theta: Optional[float] = None) -> bool:
        half = self.world_size / 2.0
        if abs(float(pos[0])) > half - self.robot_radius or abs(float(pos[1])) > half - self.robot_radius:
            return False
        for wall in self.walls:
            if self._circle_overlaps_aabb(pos, self.robot_radius, wall):
                return False
        candidate_theta = self.theta if theta is None else float(theta)
        for fork_pos, fork_length, fork_width, fork_theta in self._fork_obbs(pos, candidate_theta):
            for wall in self.walls:
                if self._obb_overlaps_aabb(
                    fork_pos, fork_length, fork_width, fork_theta, wall, margin=0.01,
                ):
                    return False
        for box in self.boxes:
            if box.delivered or box.carried or box is carried:
                continue
            if self._circle_overlaps_obb(
                pos, self.robot_radius + 0.03, box.pos, box.width, box.depth, box.theta,
            ):
                return False
        for rec in self.extras.values():
            rec_radius = rec.radius or self.robot_radius
            if self._circles_overlap(pos, self.robot_radius, rec.pos, rec_radius):
                return False
            for fork_pos, fork_length, fork_width, fork_theta in self._fork_obbs(
                pos, candidate_theta,
            ):
                if self._circle_overlaps_obb(
                    rec.pos, rec_radius + 0.02,
                    fork_pos, fork_length, fork_width, fork_theta,
                ):
                    return False
        if carried is not None:
            carry_theta = self.theta if theta is None else float(theta)
            cpos = self._carried_position(pos, carry_theta)
            for wall in self.walls:
                if self._obb_overlaps_aabb(
                    cpos, carried.width, carried.depth, carry_theta, wall, margin=0.01,
                ):
                    return False
            for box in self.boxes:
                if box.delivered or box.carried or box is carried:
                    continue
                if self._obb_overlaps_obb(
                    cpos, carried.width, carried.depth, carry_theta,
                    box.pos, box.width, box.depth, box.theta, margin=0.01,
                ):
                    return False
            for rec in self.extras.values():
                rec_radius = rec.radius or self.robot_radius
                if self._circle_overlaps_obb(
                    rec.pos, rec_radius + 0.02, cpos,
                    carried.width, carried.depth, carry_theta,
                ):
                    return False
        return True

    def _carried_position(self, pos: Optional[Sequence[float]] = None, theta: Optional[float] = None) -> np.ndarray:
        p = self.pos if pos is None else pos
        t = self.theta if theta is None else float(theta)
        reach = self.robot_radius + 0.45
        return np.asarray([float(p[0]) + math.cos(t) * reach,
                           float(p[1]) + math.sin(t) * reach], dtype=np.float32)

    def _goal_contains(self, point: Sequence[float], box: Optional[ForkliftBox] = None) -> bool:
        # A box must fit inside the goal rectangle, rather than merely touching
        # its centre, which makes stacking and delivery unambiguous.
        # A small tolerance accounts for the forklift's fixed fork reach: a
        # controller can stop when the chassis reaches the goal edge while the
        # carried box centre is a few centimetres farther forward.
        tolerance = 0.12
        if box is not None:
            corners = self._obb_corners(box.pos, box.width, box.depth, box.theta)
            x_min = float(self.goal_area_center[0] - self.goal_area_size[0] / 2.0 - tolerance)
            x_max = float(self.goal_area_center[0] + self.goal_area_size[0] / 2.0 + tolerance)
            y_min = float(self.goal_area_center[1] - self.goal_area_size[1] / 2.0 - tolerance)
            y_max = float(self.goal_area_center[1] + self.goal_area_size[1] / 2.0 + tolerance)
            return bool(
                np.all(corners[:, 0] >= x_min) and np.all(corners[:, 0] <= x_max)
                and np.all(corners[:, 1] >= y_min) and np.all(corners[:, 1] <= y_max)
            )
        return (
            abs(float(point[0]) - float(self.goal_area_center[0])) <= self.goal_area_size[0] / 2.0 + tolerance
            and abs(float(point[1]) - float(self.goal_area_center[1])) <= self.goal_area_size[1] / 2.0 + tolerance
        )

    # ------------------------------------------------------------------
    # Pickup, delivery, and step
    # ------------------------------------------------------------------
    def _can_pick(self, box: ForkliftBox) -> bool:
        available = self._available_boxes()
        if box.delivered or box.carried or not any(candidate is box for candidate in available):
            return False
        delta = box.pos - self.pos
        distance = float(np.linalg.norm(delta))
        if distance > self.pickup_distance:
            return False
        if distance < 0.60:
            # A close, centred contact is a valid fork approach regardless of
            # small heading errors from a controller.
            return True
        forward = float(delta[0] * math.cos(self.theta) + delta[1] * math.sin(self.theta))
        lateral = abs(float(-delta[0] * math.sin(self.theta) + delta[1] * math.cos(self.theta)))
        return (
            forward >= self.robot_radius * 0.30
            and lateral <= max(box.width * 1.0, 0.90)
        )

    def _try_pickup(self, requested: Optional[str] = None) -> Optional[ForkliftBox]:
        if self.carried_box_id is not None or self.fork_height > self.pickup_height + 1e-6:
            return None
        candidates = self._available_boxes()
        if requested is not None:
            candidates = [b for b in candidates if b.id == str(requested)]
        candidates.sort(key=lambda b: (float(np.linalg.norm(b.pos - self.pos)), b.id))
        for box in candidates:
            if self._can_pick(box):
                box.picked = True
                box.carried = True
                self.carried_box_id = box.id
                box.pos = self._carried_position()
                box.theta = self.theta
                box.z = self.fork_height
                self._update_goal()
                return box
        return None

    def pickup(self, box_id: Optional[str] = None) -> bool:
        """Attempt an immediate pickup for scripted controllers.

        Normal agents use the positive lift component of ``step``.  This
        helper is useful for demonstrations and preserves the same geometry and
        stack rules as the action path.
        """
        return self._try_pickup(box_id) is not None

    def _try_deliver(self, lift_command: float) -> Optional[ForkliftBox]:
        box = self.carried_box
        if box is None:
            return None
        if not self._goal_contains(box.pos, box):
            return None
        # Lowering is physical: a negative command at a high fork position
        # starts lowering but cannot release the load in mid-air.
        if self.fork_height > self.drop_height + 1e-6:
            return None
        box.carried = False
        box.delivered = True
        box.goal_level = sum(1 for b in self.boxes if b.delivered) - 1
        # Keep the destination stack aligned.  A carried pallet follows the
        # forklift while turning, but once it is placed the next pallet rests
        # on the same stack heading instead of making a twisted pile.
        delivered_stack = [other for other in self.boxes if other.delivered]
        if delivered_stack:
            box.theta = delivered_stack[0].theta
        else:
            box.theta = self.theta
        box.pos = self._goal_stack_position(box.goal_level)
        box.z = sum(b.height for b in self.boxes if b.delivered and b is not box)
        self.carried_box_id = None
        self._update_goal()
        return box

    def _goal_stack_position(self, level: int) -> np.ndarray:
        # Keep the physical centres coincident so the state really is a stack;
        # use a tiny display offset only in render().
        return self.goal_area_center.copy()

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size < 3:
            a = np.pad(a, (0, 3 - a.size))
        a = np.clip(a[:3], -1.0, 1.0).astype(np.float32)
        self._last_action[:] = a

        old_goal_kind = self._goal_kind
        old_goal_id = self.target_box_id
        old_dist = self._distance_to_goal()
        self._last_collision = False

        # Low forks automatically engage a box on contact.  The lift is
        # integrated after this check so one intuitive [0, 0, +1] action can
        # both engage the forks and begin raising a box; setting auto_pickup
        # to False restores an explicit positive lift command requirement.
        picked = self._try_pickup() if self.auto_pickup or a[2] > 0.08 else None

        self.vel[0] = float(a[0]) * self.max_linear_speed
        self.vel[1] = float(a[1]) * self.max_angular_speed
        new_theta = float(self.theta + self.vel[1] * self.dt)
        new_pos = self.pos + np.array([
            math.cos(self.theta) * self.vel[0] * self.dt,
            math.sin(self.theta) * self.vel[0] * self.dt,
        ], dtype=np.float32)
        carried = self.carried_box
        if not self._position_valid(new_pos, carried, new_theta):
            self._last_collision = True
            # Preserve turning in place when the body hits a wall, but remove
            # forward motion for the rest of the step.
            self.vel[0] = 0.0
            new_pos = self.pos.copy()
        self.pos = np.asarray(new_pos, dtype=np.float32)
        self.theta = (new_theta + math.pi) % (2.0 * math.pi) - math.pi

        self.fork_height = float(np.clip(
            self.fork_height + float(a[2]) * self.fork_speed * self.dt,
            0.0, self.max_fork_height,
        ))
        carried = self.carried_box
        if carried is not None:
            carried.pos = self._carried_position()
            carried.theta = self.theta
            carried.z = self.fork_height
        elif self.auto_pickup or a[2] > 0.08:
            picked = picked or self._try_pickup()

        delivered = self._try_deliver(float(a[2]))
        goal_switched = old_goal_kind != self._goal_kind or old_goal_id != self.target_box_id
        self._steps += 1
        self._trail.append(self.pos.tolist())
        if len(self._trail) > 300:
            self._trail = self._trail[-300:]

        # Re-targeting after a pickup/delivery changes the meaning of the goal,
        # so do not grant a fake distance jump across the task transition.
        if picked is not None or delivered is not None or goal_switched:
            progress = 0.0
            self._prev_dist = self._distance_to_goal()
        else:
            now_dist = self._distance_to_goal()
            progress = self._prev_dist - now_dist
            self._prev_dist = now_dist

        control_cost = self.action_penalty * float(np.dot(a, a))
        reward = self.progress_reward_scale * progress - self.time_penalty - control_cost
        if self._last_collision:
            reward -= self.collision_penalty
        if picked is not None:
            reward += self.pickup_reward
        if delivered is not None:
            reward += self.delivery_reward
            if self.all_boxes_delivered:
                reward += self.goal_reward

        self._last_reward = float(reward)
        self._episode_reward += self._last_reward
        terminated = self.all_boxes_delivered
        truncated = bool(self._steps >= self.max_steps and not terminated)
        self._step_extra_agents()
        info = self._info()
        info.update({
            "progress": float(progress),
            "collision": bool(self._last_collision),
            "picked_box": None if picked is None else picked.id,
            "delivered_box": None if delivered is None else delivered.id,
            "goal_switched": bool(goal_switched),
            "is_success": bool(terminated),
        })
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------
    # Observation and lidar
    # ------------------------------------------------------------------
    def _obstacle_walls(self) -> list[Wall]:
        return list(self.walls)

    def _obstacle_boxes(self) -> list[ForkliftBox]:
        return [box for box in self.boxes if not box.delivered and not box.carried]

    def _boundary_ray_distance(self, px: float, py: float, dx: float, dy: float) -> float:
        half = self.world_size / 2.0 - self.robot_radius
        ts = []
        if dx > 1e-9:
            ts.append((half - px) / dx)
        elif dx < -1e-9:
            ts.append((-half - px) / dx)
        if dy > 1e-9:
            ts.append((half - py) / dy)
        elif dy < -1e-9:
            ts.append((-half - py) / dy)
        positives = [t for t in ts if t >= 0.0]
        return min(positives) if positives else self.lidar_range

    def _lidar(self) -> np.ndarray:
        out = np.full(self.lidar_num_rays, self.lidar_range, dtype=np.float32)
        for i in range(self.lidar_num_rays):
            angle = self.theta + 2.0 * math.pi * i / self.lidar_num_rays
            dx, dy = math.cos(angle), math.sin(angle)
            best = min(self.lidar_range, self._boundary_ray_distance(float(self.pos[0]), float(self.pos[1]), dx, dy))
            for wall in self._obstacle_walls():
                # Inflate the ray geometry by the forklift body radius so a
                # reading reaches zero at the same contact point as collision.
                inflated = Wall(wall.x_lo - self.robot_radius, wall.y_lo - self.robot_radius,
                                wall.x_hi + self.robot_radius, wall.y_hi + self.robot_radius)
                best = min(best, _ray_aabb_t(float(self.pos[0]), float(self.pos[1]), dx, dy, inflated, self.lidar_range))
            # The forks are narrower than the chassis but extend farther in the
            # heading direction, so include their OBBs in the same sensor.
            for fork_pos, fork_length, fork_width, fork_theta in self._fork_obbs():
                fork = ForkliftBox(
                    id="__fork_sensor__", pos=fork_pos,
                    width=fork_length, depth=fork_width, theta=fork_theta,
                )
                best = min(
                    best,
                    self._ray_obb_t(
                        float(self.pos[0]), float(self.pos[1]), dx, dy,
                        fork, 0.02, self.lidar_range,
                    ),
                )
            for box in self._obstacle_boxes():
                best = min(
                    best,
                    self._ray_obb_t(
                        float(self.pos[0]), float(self.pos[1]), dx, dy,
                        box, self.robot_radius, self.lidar_range,
                    ),
                )
            for rec in self.extras.values():
                rec_radius = rec.radius or self.robot_radius
                best = min(
                    best,
                    self._ray_circle_t(
                        float(self.pos[0]), float(self.pos[1]), dx, dy,
                        rec.pos, rec_radius + self.robot_radius, self.lidar_range,
                    ),
                )
            out[i] = float(np.clip(best, 0.0, self.lidar_range))
        return out

    def _get_obs(self) -> np.ndarray:
        lidar = self._lidar()
        dx, dy = float(self.goal[0] - self.pos[0]), float(self.goal[1] - self.pos[1])
        c, s = math.cos(self.theta), math.sin(self.theta)
        goal_forward = dx * c + dy * s
        goal_left = -dx * s + dy * c
        remaining_fraction = float(self.remaining_count / max(1, len(self.boxes)))
        kind = {"box": 0.0, "delivery": 1.0, "complete": 2.0}[self._goal_kind]
        extra = np.array([
            goal_forward, goal_left, float(self.vel[0]), float(self.vel[1]),
            self.fork_height, 1.0 if self.carried_box_id is not None else 0.0,
            remaining_fraction, kind,
        ], dtype=np.float32)
        obs = np.concatenate([lidar, extra])
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        return {
            "goal": self.goal.copy(),
            "goal_area": self.goal_area_center.copy(),
            "goal_area_size": self.goal_area_size.copy(),
            "goal_kind": self._goal_kind,
            "target_box": self.target_box_id,
            "carried_box": self.carried_box_id,
            "pos": self.pos.copy(),
            "theta": float(self.theta),
            "fork_height": float(self.fork_height),
            "distance_to_goal": self._distance_to_goal(),
            "delivered_count": self.delivered_count,
            "remaining_boxes": self.remaining_count,
            "boxes": [box.to_dict() for box in self.boxes],
            "agents": {
                agent_id: {
                    "pos": rec.pos.copy(),
                    "theta": float(rec.theta),
                    "vel": rec.vel.copy(),
                    "goal": rec.goal.copy(),
                    "collided": bool(rec.last_collided),
                }
                for agent_id, rec in self.extras.items()
            },
            "step": self._steps,
            "episode_reward": float(self._episode_reward),
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self, view: str = "user") -> object:
        if view not in ("user", "robot"):
            view = "user"
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Arc, Circle, Polygon, Rectangle

        half = self.world_size / 2.0
        fig = Figure(figsize=(9, 9), dpi=120)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        if view == "robot":
            ax.set_xlim(float(self.pos[0]) - self.lidar_range, float(self.pos[0]) + self.lidar_range)
            ax.set_ylim(float(self.pos[1]) - self.lidar_range, float(self.pos[1]) + self.lidar_range)
        else:
            ax.set_xlim(-half, half)
            ax.set_ylim(-half, half)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        # A single low sun direction keeps shadows consistent across the
        # warehouse, boxes, and forklift in this bird's-eye view.  Every
        # frame applies the same offset to the current object positions, so
        # the light stays fixed while shadows follow moving loads.
        sun_offset = self.shadow_offset
        grid = self.world_size / 40.0
        for g in np.arange(-half, half + grid * 0.5, grid):
            ax.plot([-half, half], [g, g], color="#eeeeee", linewidth=0.4, zorder=0)
            ax.plot([g, g], [-half, half], color="#eeeeee", linewidth=0.4, zorder=0)
        ax.add_patch(Rectangle((-half, -half), self.world_size, self.world_size,
                               fill=False, edgecolor="black", linewidth=2.0, zorder=1))
        hatch = get_wall_hatch()
        for wall in self.walls:
            ax.add_patch(Rectangle(
                (wall.x_lo + sun_offset[0], wall.y_lo + sun_offset[1]),
                wall.x_hi - wall.x_lo, wall.y_hi - wall.y_lo,
                facecolor="#626b63", edgecolor="none", alpha=0.14, zorder=1.7,
            ))
            ax.add_patch(Rectangle((wall.x_lo, wall.y_lo), wall.x_hi - wall.x_lo, wall.y_hi - wall.y_lo,
                                   facecolor="white", edgecolor="#222222", linewidth=1.5,
                                   hatch=hatch, zorder=2))
        if len(self._trail) > 1:
            ax.plot([p[0] for p in self._trail], [p[1] for p in self._trail],
                    color="#86c8ef", linewidth=1.0, alpha=0.7, zorder=1.5)

        # Goal area and active target marker.
        gx, gy = map(float, self.goal_area_center)
        ax.add_patch(Rectangle(
            (gx - self.goal_area_size[0] / 2.0 + sun_offset[0],
             gy - self.goal_area_size[1] / 2.0 + sun_offset[1]),
            float(self.goal_area_size[0]), float(self.goal_area_size[1]),
            facecolor="#647968", edgecolor="none", alpha=0.10, zorder=2.05,
        ))
        ax.add_patch(Rectangle((gx - self.goal_area_size[0] / 2.0, gy - self.goal_area_size[1] / 2.0),
                               float(self.goal_area_size[0]), float(self.goal_area_size[1]),
                               facecolor="#caffd0", edgecolor="#159447", linewidth=2.0,
                               alpha=0.55, zorder=2.2))
        ax.text(gx, gy, "GOAL", ha="center", va="center", fontsize=8,
                color="#18743a", fontweight="bold", zorder=3)
        if self._goal_kind == "box":
            target = self._find_box(self.target_box_id)
            if target is not None:
                ax.add_patch(Circle(tuple(target.pos), 0.62, fill=False,
                                    edgecolor="#f28c28", linestyle="--", linewidth=1.5, zorder=4))

        # Only the exposed box in a stack gets a label.  The top source box is
        # the highest remaining stack level; at the goal, the most recently
        # delivered box is the exposed top level.  A carried box is separate
        # from both stacks and keeps its own label.
        visible_labels: set[str] = {box.id for box in self.boxes if box.carried}
        delivered_boxes = [box for box in self.boxes if box.delivered]
        if delivered_boxes:
            visible_labels.add(
                max(delivered_boxes, key=lambda box: (box.goal_level, box.id)).id
            )
        source_groups: list[list[ForkliftBox]] = []
        for box in self.boxes:
            if box.delivered or box.carried:
                continue
            group = next(
                (items for items in source_groups
                 if self._same_xy(items[0].source_pos, box.source_pos)),
                None,
            )
            if group is None:
                source_groups.append([box])
            else:
                group.append(box)
        for group in source_groups:
            visible_labels.add(
                max(group, key=lambda box: (box.stack_level, box.id)).id
            )

        def draw_box_bev(box: ForkliftBox, center: Sequence[float], *,
                         face_color: str, edge_color: str,
                         zorder: float) -> None:
            """Draw a flat, beveled BEV box with a directional sun shadow."""
            cx, cy = float(center[0]), float(center[1])
            width = max(float(box.width), 0.10)
            depth = max(float(box.depth), 0.10)
            cut = min(width, depth) * 0.14
            box_cos, box_sin = math.cos(box.theta), math.sin(box.theta)

            def rounded_rect(point: Sequence[float], scale: float = 1.0) -> list[tuple[float, float]]:
                """Return a smooth rounded rectangle in world coordinates."""
                px, py = float(point[0]), float(point[1])
                half_w = width * scale / 2.0
                half_d = depth * scale / 2.0
                radius = min(cut * scale, half_w * 0.45, half_d * 0.45)
                # Build the rounded rectangle around a local origin, then
                # rotate it around the pallet centre.  Keeping the centre out
                # of the local vertices avoids translating it twice.
                x0, x1 = -half_w, half_w
                y0, y1 = -half_d, half_d
                local_points: list[tuple[float, float]] = []
                arc_samples = 5
                arcs = (
                    ((x1 - radius, y0 + radius), -90.0, 0.0),
                    ((x1 - radius, y1 - radius), 0.0, 90.0),
                    ((x0 + radius, y1 - radius), 90.0, 180.0),
                    ((x0 + radius, y0 + radius), 180.0, 270.0),
                )
                local_points.append((x0 + radius, y0))
                local_points.append((x1 - radius, y0))
                for (center_x, center_y), start, end in arcs:
                    for angle in np.linspace(start, end, arc_samples)[1:]:
                        radians = math.radians(float(angle))
                        local_points.append((
                            center_x + radius * math.cos(radians),
                            center_y + radius * math.sin(radians),
                        ))
                return [
                    (px + box_cos * local_x - box_sin * local_y,
                     py + box_sin * local_x + box_cos * local_y)
                    for local_x, local_y in local_points
                ]

            # The sun falls down-right in the BEV image.  A broad translucent
            # offset and a smaller core keep the shadow soft without blurring
            # the vector render.
            ax.add_patch(Polygon(
                rounded_rect((cx + sun_offset[0], cy + sun_offset[1]), 1.08),
                closed=True, facecolor="#26332a", edgecolor="none",
                alpha=0.16, zorder=zorder - 0.20,
            ))
            ax.add_patch(Polygon(
                rounded_rect((cx + sun_offset[0] * 0.55, cy + sun_offset[1] * 0.55), 1.03),
                closed=True, facecolor="#1d271f", edgecolor="none",
                alpha=0.14, zorder=zorder - 0.19,
            ))
            ax.add_patch(Polygon(
                rounded_rect((cx, cy)), closed=True,
                facecolor=face_color, edgecolor=edge_color, linewidth=1.6,
                joinstyle="round", zorder=zorder,
            ))
            if box.id in visible_labels:
                ax.text(cx, cy, box.id, ha="center", va="center", fontsize=7,
                        color="#17330e", fontweight="bold", zorder=zorder + 0.1)

        # Draw bottom levels before top levels so a stack reads as layered.
        def render_key(box: ForkliftBox) -> tuple[int, int]:
            if box.carried:
                return (2, 0)
            if box.delivered:
                return (1, max(0, box.goal_level))
            return (0, max(0, box.stack_level))

        for box in sorted(self.boxes, key=render_key):
            if box.delivered:
                level = max(0, box.goal_level)
                draw_pos = self.goal_area_center.copy()
                face_color, edge_color = "#91d777", "#3e7f2d"
            elif box.carried:
                draw_pos = box.pos
                face_color, edge_color = "#91eb3e", "#3f7f1f"
            else:
                level = max(0, box.stack_level)
                draw_pos = box.pos.copy()
                face_color, edge_color = "#8ced3f", "#3e811d"
            draw_box_bev(
                box, draw_pos, face_color=face_color, edge_color=edge_color,
                zorder=6.5 + 0.01 * render_key(box)[1],
            )

        def draw_control_arcs(
            center_x: float,
            center_y: float,
            values: Sequence[float],
            *,
            radius: float,
            zorder: float,
        ) -> None:
            """Draw compact signed linear/angular control arcs.

            The two tracks sit below the object's midpoint, echoing a small
            gauge around its lower rim.  Positive values fill toward the
            lower-right side and negative values toward the lower-left side;
            the faint full tracks keep a zero command visible without labels
            or numbers.
            """
            base_angle = 270.0
            max_sweep = 66.0
            colours = ("#b36cf0", "#f08acb")
            track_alpha = 0.18
            fill_alpha = 0.86
            for index, value in enumerate(
                np.asarray(values, dtype=np.float32).reshape(-1)[:2]
            ):
                value_float = float(np.clip(value, -1.0, 1.0))
                arc_radius = max(0.18, float(radius) * (1.03 + 0.13 * index))
                bounds = (
                    float(center_x),
                    float(center_y),
                    arc_radius * 2.0,
                    arc_radius * 2.0,
                )
                track = Arc(
                    bounds[:2], bounds[2], bounds[3],
                    theta1=base_angle - max_sweep,
                    theta2=base_angle + max_sweep,
                    color=colours[index], linewidth=3.0,
                    alpha=track_alpha, zorder=zorder,
                )
                ax.add_patch(track)
                sweep = max_sweep * abs(value_float)
                if sweep <= 1e-4:
                    continue
                if value_float >= 0.0:
                    theta1, theta2 = base_angle, base_angle + sweep
                else:
                    theta1, theta2 = base_angle - sweep, base_angle
                filled = Arc(
                    bounds[:2], bounds[2], bounds[3],
                    theta1=theta1, theta2=theta2,
                    color=colours[index], linewidth=5.0,
                    alpha=fill_alpha, zorder=zorder + 0.1,
                )
                ax.add_patch(filled)

        def draw_forklift_control_lines(
            values: Sequence[float], *, zorder: float
        ) -> None:
            """Draw straight controls in the chassis' local lower-left area."""
            forward = np.asarray((c, s), dtype=np.float32)
            local_left = np.asarray((px, py), dtype=np.float32)
            # Keep the indicator inside the rear/left quadrant of the body.
            anchor = (
                self.pos
                - forward * (body_l * 0.22)
                - local_left * (body_w * 0.18)
            )
            colours = ("#b36cf0", "#f08acb")
            half_length = max(0.12, body_l * 0.20)
            row_offsets = (-0.045, 0.045)
            for index, value in enumerate(
                np.asarray(values, dtype=np.float32).reshape(-1)[:2]
            ):
                value_float = float(np.clip(value, -1.0, 1.0))
                origin = anchor + local_left * row_offsets[index]
                track_start = origin - forward * half_length
                track_end = origin + forward * half_length
                ax.plot(
                    [track_start[0], track_end[0]],
                    [track_start[1], track_end[1]],
                    color=colours[index], linewidth=3.0, alpha=0.18,
                    solid_capstyle="round", zorder=zorder,
                )
                fill_end = origin + forward * half_length * value_float
                ax.plot(
                    [origin[0], fill_end[0]], [origin[1], fill_end[1]],
                    color=colours[index], linewidth=5.0, alpha=0.86,
                    solid_capstyle="round", zorder=zorder + 0.1,
                )

        # Optional background robots.  Their bodies cast the same directional
        # shadow as the forklift, while their heading marks stay crisp and
        # shadow-free so orientation remains easy to read.
        for rec in self.extras.values():
            radius = float(rec.radius or self.robot_radius)
            if rec.color and len(rec.color) >= 2:
                body_color, edge_color = rec.color[-1], rec.color[0]
            else:
                body_color, edge_color = "#b9d8f4", "#2d5f87"
            ax.add_patch(Circle(
                (float(rec.pos[0]) + sun_offset[0], float(rec.pos[1]) + sun_offset[1]),
                radius * 1.05, facecolor="#1d2720", edgecolor="none",
                alpha=0.13, zorder=5.15,
            ))
            ax.add_patch(Circle(
                (float(rec.pos[0]) + sun_offset[0] * 0.32,
                 float(rec.pos[1]) + sun_offset[1] * 0.32),
                radius * 1.01, facecolor="#172019", edgecolor="none",
                alpha=0.20, zorder=5.25,
            ))
            ax.add_patch(Circle(
                tuple(rec.pos), radius, facecolor=body_color,
                edgecolor=edge_color, linewidth=1.6, zorder=5.6,
            ))
            fx, fy = math.cos(float(rec.theta)), math.sin(float(rec.theta))
            ax.plot(
                [float(rec.pos[0]), float(rec.pos[0]) + fx * radius],
                [float(rec.pos[1]), float(rec.pos[1]) + fy * radius],
                color="#1f2830", linewidth=1.4, zorder=5.8,
            )
            label_x = float(rec.pos[0]) - radius
            label_y = float(rec.pos[1]) + radius + 0.10
            ax.text(
                label_x, label_y, rec.id,
                ha="left", va="bottom", fontsize=6, color="#17354b",
                fontweight="bold", zorder=7.1,
            )
            draw_control_arcs(
                float(rec.pos[0]), float(rec.pos[1]), rec.last_action,
                radius=radius, zorder=7.0,
            )

        # Forklift chassis and forks, oriented by theta.  The small offset
        # shadow gives both parts a shallow 3-D lift while keeping the same
        # flat CAD style as the rest of the scene.
        c, s = math.cos(self.theta), math.sin(self.theta)
        px, py = -s, c
        body_l, body_w = self.robot_radius * 2.0, self.robot_radius * 1.55

        def body_corners(center: Sequence[float]) -> list[tuple[float, float]]:
            corners = []
            for along, side in ((-body_l / 2, -body_w / 2), (body_l / 2, -body_w / 2),
                                (body_l / 2, body_w / 2), (-body_l / 2, body_w / 2)):
                corners.append((float(center[0] + c * along + px * side),
                                float(center[1] + s * along + py * side)))
            return corners

        # Rounded joins give the chassis a small visual corner radius without
        # changing its collision radius.  The orientation circle and heading
        # line are omitted because the fork direction already shows heading.
        # Two translucent layers make the forklift shadow read as a cast
        # shadow, rather than as a second dark forklift silhouette.
        ax.add_patch(Polygon(
            body_corners(self.pos + sun_offset), closed=True,
            facecolor="#1d2720", edgecolor="none", linewidth=0.0,
            alpha=0.13, joinstyle="round", zorder=5.35,
        ))
        ax.add_patch(Polygon(
            body_corners(self.pos + sun_offset * 0.32), closed=True,
            facecolor="#172019", edgecolor="none", linewidth=0.0,
            alpha=0.20, joinstyle="round", zorder=5.45,
        ))
        ax.add_patch(Polygon(
            body_corners(self.pos), closed=True,
            facecolor="#5d6872", edgecolor="#27313b", linewidth=1.8,
            joinstyle="round", zorder=6,
        ))
        fork_base = self.robot_radius * 0.8
        # Forks enter the load and extend a little past its front edge.  That
        # exposed rounded tip keeps the heading readable while the lower
        # z-order still places the load on top of the forks.
        fork_tip = self.robot_radius + 1.15
        for side in (-1, 1):
            x0 = self.pos[0] + c * fork_base + px * side * body_w * 0.33
            y0 = self.pos[1] + s * fork_base + py * side * body_w * 0.33
            x1 = self.pos[0] + c * fork_tip + px * side * body_w * 0.33
            y1 = self.pos[1] + s * fork_tip + py * side * body_w * 0.33
            ax.plot([x0 + sun_offset[0], x1 + sun_offset[0]],
                    [y0 + sun_offset[1], y1 + sun_offset[1]],
                    color="#1d2720", linewidth=5.4, alpha=0.13,
                    solid_capstyle="round", zorder=3.65)
            ax.plot([x0 + sun_offset[0] * 0.32, x1 + sun_offset[0] * 0.32],
                    [y0 + sun_offset[1] * 0.32, y1 + sun_offset[1] * 0.32],
                    color="#172019", linewidth=4.4, alpha=0.20,
                    solid_capstyle="round", zorder=3.75)
            ax.plot([x0, x1], [y0, y1], color="#5d6872", linewidth=3.0,
                    solid_capstyle="round", zorder=4.0)

        # Keep the forklift's two driving controls fixed inside the chassis'
        # lower-left quadrant.  Lift is intentionally omitted because the
        # fork height is already shown in the title.
        draw_forklift_control_lines(self._last_action[:2], zorder=8.0)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#bbbbbb")
            spine.set_linewidth(0.8)
        ax.set_title(
            f"Forklift | {view} view | step {self._steps:03d} | "
            f"goal {self._goal_label()} | delivered {self.delivered_count}/{len(self.boxes)} | "
            f"fork {self.fork_height:.2f} m | reward {self._last_reward:+.3f}",
            fontsize=10, fontweight="bold", pad=10,
        )
        fig.tight_layout()
        self._last_fig = fig
        if self._render_mode_active == "rgb_array":
            canvas.draw()
            return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
        if self._render_mode_active == "human":
            try:
                plt.show(block=False)
            except Exception:
                pass
        return fig

    def close(self) -> None:
        if self._last_fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._last_fig)
            self._last_fig = None


def register_forklift() -> None:
    """Register ``Forklift-v0`` with Gymnasium, idempotently."""
    try:
        registry = gym.registry
        if "Forklift-v0" in registry:
            return
        gym.register(
            id="Forklift-v0",
            entry_point="forklift_gym.forklift_env:ForkliftEnv",
            max_episode_steps=700,
        )
    except Exception:
        # Registration is a convenience; direct class imports remain usable if
        # a host application has a non-standard Gymnasium registry.
        return


ForkliftEnvironment = ForkliftEnv

__all__ = ["ForkliftBox", "ForkliftEnv", "ForkliftEnvironment", "register_forklift"]
