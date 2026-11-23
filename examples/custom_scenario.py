"""One serializable scene with racks, a source stack, and scripted traffic."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image
from forklift_gym import Map, Wall, Scenario, DynamicAgentSpec, SB3ForkliftEnv


def main():
    scene = Scenario(
        map=Map(name='classroom_warehouse', world_size=12,
                walls=[Wall(-1.5,0,1.5,.3)], spawn=[-4,-4,0], goal=[4,4]),
        boxes=({'id':'bottom','pos':[-2,2],'stack_level':0},
               {'id':'top','pos':[-2,2],'stack_level':1},
               {'id':'pallet_c','pos':[2,1]}, {'id':'pallet_d','pos':[0,-1.5]}),
        goal_area={'center':[4,4], 'size':[2.4,2.4]},
        dynamic_agents=(DynamicAgentSpec('traffic_0',(-2,-2.8),0,(3,-2.8)),
                        DynamicAgentSpec('traffic_1',(2.5,3.5),math.pi,(-3,3.5))),
    )
    output = Path('outputs'); output.mkdir(exist_ok=True)
    (output/'scenario.json').write_text(json.dumps(scene.to_dict(), indent=2)+'\n')
    env = SB3ForkliftEnv(scenario=scene, render_mode='rgb_array')
    try:
        env.reset(seed=7)
        for _ in range(30):
            env.step(np.zeros(3, dtype=np.float32))
        Image.fromarray(env.render()).save(output/'custom_scenario.png')
    finally:
        env.close()


if __name__ == '__main__':
    main()
