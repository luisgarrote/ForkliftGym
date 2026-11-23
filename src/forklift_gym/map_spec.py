"""
Map model + named preset store for the mobile robot gym.

A *map* is everything that defines an arena's static scene:

    * ``world_size``  -- the (square) arena extent, ``[-H, H]^2`` with ``H = size/2``
    * ``walls``       -- the list of axis-aligned (AABB) ``Wall`` rectangles
    * ``spawn``       -- the focus robot's start point ``[x, y, theta]``
    * ``goal``        -- the focus goal ``[x, y]``
    * ``elements``    -- free-form *drawable* objects the map can carry beyond
      walls: geometric ``corners``, labelled ``markers``, and ``object`` boxes
      ("undefined objects" -- generic obstacles / placeholders; optionally
      collidable so they block the robot like a wall).

Maps are JSON-serializable.  Presets are stored, one file per preset, in the
``maps/`` directory next to this module.  ``MobileRobotEnv(map=...)`` accepts a
``Map``, a dict, a path, or a preset name, so a map edited on disk drives the
environment without touching any Python source.

The built-in ``"t_corridor"`` preset reproduces the layout the environment built
before maps existed, so an env created with no map and an env created with the
default preset are byte-for-byte identical.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .primitives import Wall

# ===========================================================================
#  Drawable element -- a corner / marker / object the map can carry
# ===========================================================================
# ``kind`` is one of: "corner" (a geometric corner / reference point),
# "marker" (a labelled free point), or "object" (a generic box, optionally
# collidable -- an "undefined object" placeholder).
ELEMENT_KINDS = ("corner", "marker", "object")


@dataclass
class MapElement:
    """A free-form drawable added to a map, beyond the wall set.

    Parameters
    ----------
    kind: "corner" | "marker" | "object".
    x, y: centre (in world coords).  For an ``object`` this is the box centre.
    label: free text (drawn next to the element / used in the file's legend).
    w, h: box width / height in world units (only meaningful for "object").
    collide: for an "object", whether the box blocks the robot like a wall.
    """

    kind: str = "marker"
    x: float = 0.0
    y: float = 0.0
    label: str = ""
    w: float = 0.0
    h: float = 0.0
    collide: bool = False

    def __post_init__(self) -> None:
        self.kind = str(self.kind)
        self.label = str(self.label)
        self.x = float(self.x)
        self.y = float(self.y)
        self.w = float(self.w)
        self.h = float(self.h)
        self.collide = bool(self.collide)
        if self.kind not in ELEMENT_KINDS:
            # Tolerate unknown kinds but never crash the pipeline.
            self.kind = "object" if self.kind == "object" else "marker"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"kind": self.kind, "x": self.x, "y": self.y}
        if self.label:
            d["label"] = self.label
        if self.kind == "object":
            d["w"], d["h"] = self.w, self.h
            d["collide"] = self.collide
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MapElement":
        return cls(
            kind=d.get("kind", "marker"),
            x=float(d.get("x", 0.0)),
            y=float(d.get("y", 0.0)),
            label=d.get("label", ""),
            w=float(d.get("w", 0.0)),
            h=float(d.get("h", 0.0)),
            collide=bool(d.get("collide", False)),
        )

    @property
    def aabb(self) -> Optional[Wall]:
        """The AABB the element occupies, or ``None`` for non-box kinds."""
        if self.kind != "object" or (self.w <= 0.0 or self.h <= 0.0):
            return None
        w, h = self.w, self.h
        return Wall(
            self.x - w / 2.0, self.y - h / 2.0, self.x + w / 2.0, self.y + h / 2.0
        )

    def contains(self, px: float, py: float) -> bool:
        """True when ``(px, py)`` lies inside this element (box or near-point)."""
        if self.kind == "object":
            box = self.aabb
            if box is None:
                return False
            return box.x_lo <= px <= box.x_hi and box.y_lo <= py <= box.y_hi
        # point kinds: a small hit radius so the cursor can grab them easily
        return ((px - self.x) ** 2 + (py - self.y) ** 2) < 0.15 * 0.15

    def center(self) -> tuple[float, float]:
        return (self.x, self.y)


# ===========================================================================
#  Map -- the full static scene
# ===========================================================================
@dataclass
class Map:
    """A serializable arena: world size, walls, focus spawn/goal, and elements.

    ``spawn`` is ``[x, y, theta]`` and ``goal`` is ``[x, y]``.  ``name`` labels
    the preset when it is saved; it is not required.
    """

    name: str = "untitled"
    world_size: float = 12.0
    walls: List[Wall] = field(default_factory=list)
    spawn: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    goal: List[float] = field(default_factory=lambda: [0.0, 0.0])
    elements: List[MapElement] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.world_size = float(self.world_size)
        if len(self.spawn) == 2:
            self.spawn.append(0.0)
        if len(self.goal) == 1:
            self.goal.append(self.goal[0])

    # ------------------------------------------------------------------
    #  Collidable boxes: the walls plus any colliding "object" elements.
    # ------------------------------------------------------------------
    def collision_walls(self) -> List[Wall]:
        """Every box that should block the robot (walls + colliding objects)."""
        out = list(self.walls)
        for el in self.elements:
            if getattr(el, "collide", False):
                box = el.aabb
                if box is not None:
                    out.append(box)
        return out

    # ------------------------------------------------------------------
    #  Serialization round-trips losslessly to / from a plain dict.
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "world_size": self.world_size,
            "walls": [
                {"x_lo": w.x_lo, "y_lo": w.y_lo, "x_hi": w.x_hi, "y_hi": w.y_hi}
                for w in self.walls
            ],
            "spawn": list(self.spawn),
            "goal": list(self.goal),
            "elements": [el.to_dict() for el in self.elements],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Map":
        walls = [
            Wall(
                float(w.get("x_lo", w.get("x0"))),
                float(w.get("y_lo", w.get("y0"))),
                float(w.get("x_hi", w.get("x1"))),
                float(w.get("y_hi", w.get("y1"))),
            )
            for w in d.get("walls", [])
        ]
        return cls(
            name=d.get("name", "untitled"),
            world_size=float(d.get("world_size", 12.0)),
            walls=walls,
            spawn=list(d.get("spawn", [0.0, 0.0, 0.0]))[0:3],
            goal=list(d.get("goal", [0.0, 0.0]))[0:2],
            elements=[MapElement.from_dict(e) for e in d.get("elements", [])],
            meta=dict(d.get("meta", {})),
        )

    def save(self, path: str) -> None:
        path = os.fspath(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "Map":
        with open(os.fspath(path)) as f:
            return cls.from_dict(json.load(f))

    # ------------------------------------------------------------------
    #  Preset store -- one JSON file per preset in maps/ next to this module.
    # ------------------------------------------------------------------
    @staticmethod
    def preset_dir() -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps")

    def save_preset(self, name: str = None, directory: Optional[str] = None) -> str:
        """Write this map to ``<directory or maps/><name>.json`` and return the path."""
        name = str(name or self.name or "untitled")
        # keep the filename safe
        name = "".join(c for c in name if c not in '/\\:*?"<>|') or "untitled"
        directory = directory or self.preset_dir()
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, name + ".json")
        self.name = name
        self.save(path)
        return path

    @classmethod
    def load_preset(cls, name: str, directory: Optional[str] = None) -> "Map":
        directory = directory or cls.preset_dir()
        raw = os.path.join(directory, name + ".json")
        if os.path.exists(raw):
            m = cls.load(raw)
            m.name = name
            return m
        # fall back to a built-in preset shipped in code (factory callables)
        factory = BUILTIN_PRESETS.get(name)
        if factory is not None:
            m = factory()
            m.name = name
            return m
        raise FileNotFoundError(
            f"no preset {name!r} in {directory!r} or built-ins "
            f"({sorted(BUILTIN_PRESETS)})"
        )

    @classmethod
    def list_presets(cls, directory: Optional[str] = None) -> List[str]:
        names = set(BUILTIN_PRESETS)
        directory = directory or cls.preset_dir()
        if os.path.isdir(directory):
            for fn in os.listdir(directory):
                if fn.endswith(".json"):
                    names.add(fn[:-5])
        return sorted(names)

    @classmethod
    def default(cls) -> "Map":
        """The layout the environment shipped with (``t_corridor``)."""
        return BUILTIN_PRESETS["t_corridor"]()


# ===========================================================================
#  Built-in presets -- factory callables so each env gets a fresh copy.
# ===========================================================================
def _t_corridor() -> Map:
    """Verbatim the layout from env.reset() so default behaviour is unchanged."""
    H = 12.0 / 2.0
    t = 0.35
    walls = [
        Wall(-t / 2, -H * 0.35, t / 2, H),  # vertical spine of the T
        Wall(-H, H * 0.15 - t / 2, H * 0.85, H * 0.15 + t / 2),  # top bar
        Wall(H * 0.20, -H * 0.55, H * 0.20 + t, H * 0.15),  # right partition
        Wall(-H * 0.80, -H * 0.05, -H * 0.30, -H * 0.05 + t),  # L room wall
    ]
    return Map(
        name="t_corridor",
        world_size=12.0,
        walls=walls,
        spawn=[-H * 0.40, -H * 0.50, 0.0],
        goal=[H * 0.60, H * 0.60],
    )


def _empty_arena() -> Map:
    H = 10.0 / 2.0
    return Map(
        name="empty_arena",
        world_size=10.0,
        walls=[],
        spawn=[0.0, -H * 0.6, 0.0],
        goal=[0.0, H * 0.6],
    )


def _grid_maze() -> Map:
    """A small 3x3-ish grid with colliding object 'blocks' and a couple of
    corner markers -- demonstrates the editable element types."""
    H = 10.0 / 2.0
    s = H / 3.0
    walls = [
        # outer-ish interior cross
        Wall(-s / 2, -s, s / 2, s),
        Wall(-s, s / 2, s, s / 2 + 0.15),
    ]
    elements = [
        MapElement("object", x=-1.5, y=-1.5, w=0.6, h=0.6, collide=True, label="crate"),
        MapElement("corner", x=H, y=H, label="NE"),
        MapElement("corner", x=-H, y=-H, label="SW"),
        MapElement("marker", x=2.0, y=2.0, label="pickup"),
    ]
    return Map(
        name="grid_maze",
        world_size=10.0,
        walls=walls,
        spawn=[-H * 0.7, -H * 0.7, 0.0],
        goal=[H * 0.7, H * 0.7],
        elements=elements,
    )


BUILTIN_PRESETS: Dict[str, Any] = {
    "t_corridor": _t_corridor,
    "default": _t_corridor,
    "empty_arena": _empty_arena,
    "grid_maze": _grid_maze,
}


# ===========================================================================
#  Loader -- one entry point that turns any supported spec into a Map.
# ===========================================================================
def load_map(spec: Union[str, os.PathLike, dict, Map, None]) -> Map:
    """Coerce a map spec into a ``Map``.

    * ``None``         -> the default ``t_corridor`` layout.
    * a ``Map``        -> returned as-is.
    * a dict           -> parsed via ``Map.from_dict``.
    * a path           -> ``Map.load`` (a .json preset on disk).
    * a bare name      -> ``Map.load_preset`` (a built-in preset).
    """
    if spec is None:
        return Map.default()
    if isinstance(spec, Map):
        return spec
    if isinstance(spec, dict):
        return Map.from_dict(spec)
    p = os.fspath(spec)
    if p.endswith(".json") or os.path.exists(p):
        return Map.load(p)
    return Map.load_preset(p)


__all__ = [
    "Map",
    "MapElement",
    "Wall",
    "load_map",
    "BUILTIN_PRESETS",
    "ELEMENT_KINDS",
]
