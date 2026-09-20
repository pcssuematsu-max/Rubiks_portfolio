import tempfile
import unittest
from pathlib import Path

import numpy as np

from managers.param_manager import ParamManager


class _AI:
    def __init__(self):
        self.params = {"W1": np.zeros((1, 2), dtype = "f")}
        self.v = {"W1": np.zeros((1, 2), dtype = "f")}
        self.h = {"W1": np.zeros((1, 2), dtype = "f")}
        self.dirty_calls = 0
        self.perfect_calls = 0

    def mark_params_dirty(self):
        self.dirty_calls += 1

    def set_perfect_val(self):
        self.perfect_calls += 1


class _Frame:
    def __init__(self):
        self.AIs = [_AI()]
        self.cube = object()
        self.events = []

    def record_ai_lifecycle_event(self, event_type, ai_index = None, details = None):
        self.events.append((event_type, ai_index, details))


class ParamManagerTests(unittest.TestCase):
    def test_load_records_exactly_which_parameter_files_were_restored(self):
        frame = _Frame()
        manager = ParamManager(frame)
        with tempfile.TemporaryDirectory() as temporary_directory:
            manager.project_root = Path(temporary_directory)
            data_dir = manager._data_dir(0)
            data_dir.mkdir()
            np.save(data_dir / "W1.npy", np.asarray(((3.0, 4.0),), dtype = "f"))
            np.save(data_dir / "W1_v.npy", np.asarray(((5.0, 6.0),), dtype = "f"))
            np.save(data_dir / "W1_h.npy", np.asarray(((7.0, 8.0),), dtype = "f"))

            manager.load(0)

        self.assertTrue(np.allclose(frame.AIs[0].params["W1"], ((3.0, 4.0),)))
        self.assertEqual(frame.events, [
            (
                "parameters_loaded",
                0,
                {
                    "source": "arrayFiles",
                    "parameterDirectory": str(data_dir),
                    "requestedParameterCount": 1,
                    "loadedParameterCount": 1,
                    "loadedKeys": ["W1"],
                },
            ),
        ])


if __name__ == "__main__":
    unittest.main()
