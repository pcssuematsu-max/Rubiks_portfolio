import tempfile
import unittest
from pathlib import Path

from core.ai_lifecycle_log import AiLifecycleLogStore


class AiLifecycleLogStoreTests(unittest.TestCase):
    def test_appends_and_reads_restart_load_and_normalization_events(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "lifecycle.jsonl"
            store = AiLifecycleLogStore(path)

            store.append(
                "application_started",
                "session-a",
                details={"cubeSize": 7, "aiCount": 20},
            )
            store.append(
                "parameters_loaded",
                "session-a",
                ai_index = 10,
                details={"source": "arrayFiles", "loadedParameterCount": 42},
            )
            store.append(
                "parameters_normalized",
                "session-a",
                ai_index = 10,
                details={"normalizedRows": {"WQ1": 64}, "targetVariance": 0.125},
            )

            records = store.records()

        self.assertEqual([record["eventType"] for record in records], [
            "application_started", "parameters_loaded", "parameters_normalized",
        ])
        self.assertEqual(records[0]["aiIndex"], None)
        self.assertEqual(records[1]["aiIndex"], 10)
        self.assertEqual(records[2]["details"]["normalizedRows"]["WQ1"], 64)

    def test_rejects_non_json_safe_details(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = AiLifecycleLogStore(Path(temporary_directory) / "lifecycle.jsonl")

            with self.assertRaisesRegex(ValueError, "JSON-safe"):
                store.append("application_started", "session-a", details={"bad": float("nan")})


if __name__ == "__main__":
    unittest.main()
