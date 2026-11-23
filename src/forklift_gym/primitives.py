"""Shared wall, ray and traffic-agent primitives for the standalone forklift."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import numpy as np

class Wall:
    __slots__ = ("x_lo", "y_lo", "x_hi", "y_hi")

    def __init__(self, x_lo: float, y_lo: float, x_hi: float, y_hi: float) -> None:
        self.x_lo = min(float(x_lo), float(x_hi))
        self.y_lo = min(float(y_lo), float(y_hi))
        self.x_hi = max(float(x_lo), float(x_hi))
        self.y_hi = max(float(y_lo), float(y_hi))


def _ray_aabb_t(px: float, py: float, dx: float, dy: float,
                w: Wall, max_t: float) -> float:
    """Smallest positive intersection distance of a ray (origin ``p``) with an
    AABB wall ``w`` in max distance ``max_t``, or ``inf`` when none.
    Standard slab method; works for any robot radius by inflating the box by r.
    """
    t_min = 0.0
    t_max = max_t
    for origin, dir_c, lo, hi in (
        (px, dx, w.x_lo, w.x_hi),
        (py, dy, w.y_lo, w.y_hi),
    ):
        if abs(dir_c) < 1e-9:
            # Ray parallel to this slab — no intersection on this axis
            if origin < lo or origin > hi:
                return float("inf")
            continue
        t1 = (lo - origin) / dir_c
        t2 = (hi - origin) / dir_c
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return float("inf")
    return t_min if t_min > 0 else t_max


# ===========================================================================
#  Rendering helpers (version-robust)
# ===========================================================================
_WALL_HATCH: Optional[object] = None


def _wall_hatch() -> object:
    """A fine single-direction CAD hatch that is robust across matplotlib
    versions.  Prefer a sparse, clearly-visible diagonal hatch; fall back to a
    plain string hatch if a custom object is ever rejected.
    """
    try:
        import matplotlib
        from matplotlib.hatch import NorthEastHatch
        from matplotlib.patches import Rectangle
        import matplotlib.pyplot as plt

        obj = NorthEastHatch("/", 0.6)
        # Only use it if matplotlib actually accepts it as a hatch object.
        fig, ax = plt.subplots()
        ax.add_patch(Rectangle((0, 0), 1, 1, hatch=obj, fc="w", ec="k"))
        fig.canvas.draw()
        plt.close(fig)
        return obj
    except Exception:
        # Plain single-direction diagonal CAD hatch — version independent.
        return "///"


def get_wall_hatch() -> object:
    """Lazily build (and cache) the wall hatch used by every render."""
    global _WALL_HATCH
    if _WALL_HATCH is None:
        _WALL_HATCH = _wall_hatch()
    return _WALL_HATCH


@dataclass
class AgentRecord:
    """Live record for one keyed robot in the multi-agent registry.

    `pos` and `vel` are numpy arrays mutated in place as the agent is stepped;
    `method` is an optional external policy `method(obs, env) -> [v, omega]`.
     `radius` is this robot's collision size in metres: a value of 0.0 means
    "use the environment's default `robot_radius`".  Inter-agent separation,
    wall push-out, and lidar inflation all use each robot's own radius.
    """
    id: str
    pos: np.ndarray
    theta: float
    vel: np.ndarray
    goal: np.ndarray
    method: Optional[Callable[[np.ndarray, Any], object]] = None
    controllable: bool = True
    color: Optional[tuple] = None
    radius: float = 0.0
    default_action: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    last_collided: bool = False
    last_action: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    # Number of in-place recovery turns requested by the default patrol policy
    # after a wall/robot contact.  Custom policies may ignore this field.
    stuck_steps: int = 0
