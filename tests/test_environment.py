"""Contracts and task mechanics of the standalone public package."""
import json
import unittest
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import forklift_gym
from forklift_gym import ForkliftEnv, ForkliftParallelEnv, SB3ForkliftEnv, Scenario, Map, Wall


class ForkliftTests(unittest.TestCase):
    def make(self, **kwargs):
        env = ForkliftEnv(**kwargs)
        self.addCleanup(env.close)
        return env

    def test_registered_contract_and_determinism(self):
        env = gym.make('Forklift-v0', render_mode='rgb_array')
        self.addCleanup(env.close)
        check_env(env.unwrapped, skip_render_check=True)
        first, _ = env.reset(seed=7)
        second, _ = env.reset(seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (44,))
        self.assertTrue(env.observation_space.contains(first))
        self.assertEqual(env.action_space.shape, (3,))
        image = env.render()
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image.shape[-1], 3)

    def test_package_maps_are_installed(self):
        self.assertIn('empty_arena', Map.list_presets())
        env = self.make(map='empty_arena')
        obs, _ = env.reset(seed=7)
        self.assertEqual(obs.shape, (44,))

    def test_seeded_random_boxes_and_actions(self):
        first = self.make(num_boxes=4)
        second = self.make(num_boxes=4)
        first.reset(seed=41); second.reset(seed=41)
        for _ in range(20):
            a = np.array([.2,.1,0], dtype=np.float32)
            one = first.step(a); two = second.step(a)
            np.testing.assert_array_equal(one[0],two[0])
            self.assertEqual(one[1:4],two[1:4])

    def test_only_top_box_can_be_picked_and_it_rotates(self):
        env = self.make(walls=[], spawn=[-1,0,0],
                        boxes=[{'id':'bottom','pos':[0,0]}, {'id':'top','pos':[0,0]}])
        env.reset(seed=7)
        self.assertFalse(env.pickup('bottom'))
        _, reward, _, _, info = env.step([0,0,1])
        self.assertEqual(info['picked_box'],'top')
        self.assertEqual(info['goal_kind'],'delivery')
        self.assertEqual(info['progress'],0.0)
        self.assertGreater(reward,0)
        env.step([0,.25,1])
        self.assertAlmostEqual(env.carried_box.theta,env.theta)
        self.assertFalse(env.get_box('bottom').carried)

    def test_delivery_height_progress_and_success(self):
        env = self.make(walls=[], spawn=[-1,0,0], boxes=[{'id':'cargo','pos':[0,0]}],
                        goal_area={'center':[3,3],'size':[2.4,2.4]})
        env.reset(seed=7)
        for _ in range(8):env.step([0,0,1])
        self.assertTrue(env.holding_box)
        env.set_pos([3-(env.robot_radius+.45),3],theta=0)
        _, _, terminated, _, info = env.step([0,0,0])
        self.assertFalse(terminated)
        self.assertEqual(info['delivered_count'],0)
        for _ in range(10):
            _, _, terminated, truncated, info = env.step([0,0,-1])
            if terminated:break
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info['is_success'])
        self.assertEqual(info['delivered_box'],'cargo')
        self.assertEqual(info['progress'],0.0)
        self.assertEqual(env.delivered_count,1)

    def test_forks_participate_in_collision_geometry(self):
        wall = Wall(1.25,-.4,1.35,.4)
        env = self.make(walls=[wall])
        env.reset(seed=7)
        self.assertFalse(env._circle_overlaps_aabb([0,0],env.robot_radius,wall))
        self.assertFalse(env._position_valid([0,0],theta=0))

    def test_timeout_is_truncation(self):
        env = self.make(max_steps=1)
        env.reset(seed=7)
        _, _, terminated, truncated, _ = env.step([0,0,0])
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_traffic_adapter_contract_and_motion(self):
        env = SB3ForkliftEnv(num_dynamic_agents=2,obstacle_count=2,num_boxes=4,max_steps=100)
        self.addCleanup(env.close)
        check_env(env,skip_render_check=True)
        obs, _ = env.reset(seed=7)
        positions = {key:r.pos.copy() for key,r in env.forklift.extras.items()}
        self.assertEqual(len(positions),2)
        for _ in range(80):
            obs, _, terminated, truncated, _ = env.step([0,0,0])
            self.assertFalse(terminated or truncated)
            self.assertTrue(env.observation_space.contains(obs))
        self.assertTrue(all(np.linalg.norm(env.forklift.extras[key].pos-pos)>.1
                            for key,pos in positions.items()))

    def test_generated_episode_sequence_remains_seeded(self):
        first = SB3ForkliftEnv(num_dynamic_agents=2,obstacle_count=2,num_boxes=4)
        second = SB3ForkliftEnv(num_dynamic_agents=2,obstacle_count=2,num_boxes=4)
        self.addCleanup(first.close); self.addCleanup(second.close)
        first.reset(seed=123); second.reset(seed=123)
        for _ in range(3):
            a, _ = first.reset(); b, _ = second.reset()
            np.testing.assert_array_equal(a,b)
            for one,two in zip(first.forklift.boxes,second.forklift.boxes):
                np.testing.assert_array_equal(one.pos,two.pos)

    def test_scenario_round_trip_and_parallel_dictionary_api(self):
        scene = Scenario(map=Map(name='custom',world_size=12,walls=[],spawn=[-4,-4,0],goal=[4,4]),
                         boxes=({'id':'one','pos':[0,0]},),
                         goal_area={'center':[4,4],'size':[2,2]})
        loaded = forklift_gym.load_scenario(json.loads(json.dumps(scene.to_dict())))
        env = ForkliftParallelEnv(scenario=loaded,max_steps=1)
        self.addCleanup(env.close)
        observations, _ = env.reset(seed=7)
        self.assertEqual(list(observations),['main'])
        _, _, terminated, truncated, _ = env.step({'main':[0,0,0]})
        self.assertFalse(terminated['main'])
        self.assertTrue(truncated['main'])
        self.assertEqual(env.agents,[])


if __name__=='__main__':
    unittest.main()
