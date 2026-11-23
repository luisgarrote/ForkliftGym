"""ForkliftGym: Gymnasium material handling with optional scripted traffic."""
import gymnasium as gym
from .forklift_env import ForkliftEnv, ForkliftEnvironment, ForkliftBox, register_forklift
from .forklift_pettingzoo_env import ForkliftParallelEnv, SB3ForkliftEnv
from .primitives import Wall
from .map_spec import Map, MapElement, load_map
from .scenarios import (Scenario, DynamicAgentSpec, RandomScenario, CurriculumScenario,
                        load_scenario, make_static_scenario, goal_seek_policy)

__version__ = "0.1.0"

def register():
    register_forklift()
    if "ForkliftTraffic-v0" not in gym.registry:
        gym.register(id="ForkliftTraffic-v0",
                     entry_point="forklift_gym.forklift_pettingzoo_env:SB3ForkliftEnv")

register()

__all__ = ["ForkliftEnv", "ForkliftEnvironment", "ForkliftBox", "ForkliftParallelEnv",
           "SB3ForkliftEnv", "Wall", "Map", "MapElement", "load_map", "Scenario",
           "DynamicAgentSpec", "RandomScenario", "CurriculumScenario", "load_scenario",
           "make_static_scenario", "goal_seek_policy", "register"]
