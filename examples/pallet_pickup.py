"""Animate real pallet pickup and delivery with a small scripted controller.

This is an environment demonstration, not a trained policy. Every simulator step
runs; GIF capture samples frames, with a short pause on pickup/delivery events.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
from forklift_gym import ForkliftEnv


def control(env):
    if env.holding_box:
        box = env.carried_box
        if env._goal_contains(box.pos, box):
            return np.array([0,0,-1], dtype=np.float32), 'Lower and deliver'
        if env.fork_height < .9:
            return np.array([0,0,1], dtype=np.float32), 'Lift the pallet'
        phase = 'Carry to the delivery area'
    else:
        if env.fork_height > .05:
            return np.array([0,0,-1], dtype=np.float32), 'Lower the forks'
        phase = 'Approach the next pallet'
    delta = env.goal-env.pos
    distance = float(np.linalg.norm(delta))
    error = (math.atan2(float(delta[1]),float(delta[0]))-env.theta+math.pi)%(2*math.pi)-math.pi
    turn = np.clip(2.5*error/env.max_angular_speed,-1,1)
    drive = min(.65,.5*distance) if abs(error)<.45 else 0.0
    return np.array([drive,turn,0], dtype=np.float32), phase


def render(env, phase, width):
    env.render()
    fig = env._last_fig
    fig.axes[0].set_title(f'{phase} | Delivered {env.delivered_count}/2', fontsize=11)
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
    frame = Image.fromarray(rgb)
    frame.thumbnail((width,width))
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/pallet_pickup.gif')
    parser.add_argument('--max-steps', type=int, default=800)
    parser.add_argument('--frame-stride', type=int, default=3)
    parser.add_argument('--width', type=int, default=640)
    args = parser.parse_args()
    if min(args.max_steps,args.frame_stride,args.width)<1:
        parser.error('counts and width must be positive')
    env = ForkliftEnv(render_mode='rgb_array', spawn=[-4,-1.4,0],
                      boxes=[{'id':'pallet_A','pos':[-1.8,-1.4]},
                             {'id':'pallet_B','pos':[-1.8,1.4]}],
                      goal_area={'center':[3,2],'size':[2.4,2.4]},max_steps=args.max_steps)
    frames=[];durations=[];events=[]
    try:
        env.reset(seed=7)
        frames.append(render(env,'Approach the first pallet',args.width));durations.append(500)
        for step in range(1,args.max_steps+1):
            action, phase = control(env)
            _, _, terminated, truncated, info = env.step(action)
            event = info['picked_box'] is not None or info['delivered_box'] is not None
            if event:
                events.append({'step':step,'time_seconds':round(step*env.dt,2),
                               'picked_box':info['picked_box'],'delivered_box':info['delivered_box']})
                phase = 'Pallet caught on the forks' if info['picked_box'] else 'Pallet delivered'
            if step%args.frame_stride==0 or event or terminated:
                frames.append(render(env,phase,args.width))
                # Normal playback is twice simulation speed; event pauses aid inspection.
                durations.append(600 if event else round(1000*env.dt*args.frame_stride/2))
            if terminated or truncated:
                break
        if env.delivered_count!=2:
            raise RuntimeError(f'Demonstration incomplete after {step} steps: {env.delivered_count}/2 delivered')
        frames.append(render(env,'Both pallets delivered and stacked',args.width));durations.append(1500)
        target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
        frames[0].save(target,save_all=True,append_images=frames[1:],duration=durations,loop=0)
        metadata={'controller':'scripted','seed':7,'dt':env.dt,'steps':step,
                  'delivered':env.delivered_count,'events':events,
                  'frame_stride':args.frame_stride,'normal_playback_speed':2}
        target.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(json.dumps(metadata,indent=2))
        print(target)
    finally:
        env.close()


if __name__=='__main__':
    main()
