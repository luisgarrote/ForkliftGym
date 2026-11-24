"""Keep the student benchmark paired and reproducible as examples evolve."""
import json
from pathlib import Path
import unittest
import numpy as np
from forklift_gym import SB3ForkliftEnv, load_scenario


class FixedScenarioTests(unittest.TestCase):
    def test_five_normal_and_five_matched_traffic_cases(self):
        self.check_pairs("evaluation_cases.json", "")

    def test_validation_pairs_are_separate_from_final_tests(self):
        self.check_pairs("validation_cases.json", "val_")
        examples = Path(__file__).resolve().parents[1] / "examples"
        validation = json.loads((examples/"validation_cases.json").read_text())["cases"]
        testing = json.loads((examples/"evaluation_cases.json").read_text())["cases"]
        self.assertFalse({case["seed"] for case in validation} & {case["seed"] for case in testing})

    def check_pairs(self, filename, prefix):
        manifest = Path(__file__).resolve().parents[1] / "examples" / filename
        cases = {case['id']: case for case in json.loads(manifest.read_text())['cases']}
        self.assertEqual(len(cases), 10)
        for i in range(1, 6):
            normal, dynamic = cases[f'{prefix}normal_{i}'], cases[f'{prefix}dynamic_{i}']
            a, b = normal['scenario'], dynamic['scenario']
            self.assertEqual(normal['seed'], dynamic['seed'])
            self.assertEqual(normal['max_steps'], 700)
            for field in ('map', 'boxes', 'goal_area'):
                self.assertEqual(a[field], b[field])
            self.assertEqual(a['dynamic_agents'], [])
            self.assertIn(len(b['dynamic_agents']), (1, 2))
            environments = [SB3ForkliftEnv(scenario=load_scenario(scene)) for scene in (a, b)]
            try:
                for env in environments:
                    env.reset(seed=normal['seed'])
                first, second = [env.forklift for env in environments]
                np.testing.assert_array_equal(first.pos, second.pos)
                self.assertEqual(first.theta, second.theta)
                self.assertEqual(len(first.boxes), 4)
                self.assertEqual([box.to_dict() for box in first.boxes],
                                 [box.to_dict() for box in second.boxes])
            finally:
                for env in environments:
                    env.close()
