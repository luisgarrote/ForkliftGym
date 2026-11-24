"""Keep the compatibility command aligned with the student baseline."""
import unittest
from unittest.mock import patch
from forklift_gym import cli


class CLIDefaultTests(unittest.TestCase):
    def test_history_defaults_to_one(self):
        with patch.object(cli, "train") as train, patch(
            "sys.argv", ["forklift-lab", "train", "--output", "unused"]
        ):
            cli.main()
        train.assert_called_once()
        self.assertEqual(train.call_args.args[0].history, 1)
