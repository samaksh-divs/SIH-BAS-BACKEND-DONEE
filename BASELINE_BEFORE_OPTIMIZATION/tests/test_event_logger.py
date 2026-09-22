"""
Unit tests for Event Logger (Corrected Semantics verification)
"""
import json
import os
import shutil
import unittest
from src.event_logger import EventLogger
from src.state_machine import StateUpdate, ExperimentStateMachine
from src.action_inference import ObservedAction


class TestEventLogger(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_logs"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_step_completed_corrected_semantics(self):
        """Verifies that STEP_COMPLETED log records contain unambiguous completed vs next step fields."""
        logger = EventLogger(log_dir=self.test_dir, custom_filename="semantics_test.jsonl")

        up = StateUpdate(
            current_step=2,
            current_state_id="S02",
            expected_action="RETRIEVE_RED_BOX",
            observed_action="OPEN_WHITE_BOX",
            next_step="S03",
            status="STEP_COMPLETED",
            error_type=None,
            confidence=0.95,
            message="Step complete. Next: Retrieve the red box.",
            transitioned=True,
            previous_state="S01",
            timestamp=0.3,
            frame=3,
            completed_state="S01",
            completed_step_number=1,
            completed_action="OPEN_WHITE_BOX"
        )

        rec = logger.log_state_update(up)
        logger.close()

        self.assertIsNotNone(rec)
        self.assertEqual(rec["event"], "STEP_COMPLETED")
        self.assertEqual(rec["completed_state"], "S01")
        self.assertEqual(rec["completed_step_number"], 1)
        self.assertEqual(rec["completed_action"], "OPEN_WHITE_BOX")
        self.assertEqual(rec["next_state"], "S02")
        self.assertEqual(rec["next_step_number"], 2)
        self.assertEqual(rec["next_action"], "RETRIEVE_RED_BOX")

    def test_log_creation_and_jsonl_validity(self):
        """Tests Logger creates valid JSONL and unique files across runs."""
        logger1 = EventLogger(log_dir=self.test_dir, custom_filename="run1.jsonl")
        logger1.log_event(1.0, 10, "EXPERIMENT_STARTED", "S01", 1, "OPEN_WHITE_BOX", "OPEN_WHITE_BOX", "START", 1.0, "Started")
        logger1.close()

        logger2 = EventLogger(log_dir=self.test_dir, custom_filename="run2.jsonl")
        logger2.log_step_completed(
            timestamp=2.0, frame=20, completed_state="S01", completed_step_number=1, completed_action="OPEN_WHITE_BOX",
            next_state="S02", next_step_number=2, next_action="RETRIEVE_RED_BOX", confidence=0.9, message="Completed"
        )
        logger2.close()

        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "run1.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "run2.jsonl")))

        with open(os.path.join(self.test_dir, "run1.jsonl"), 'r', encoding='utf-8') as f:
            lines = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(lines[0]["event"], "EXPERIMENT_STARTED")

    def test_event_types_logging(self):
        """Logging experiment start, step completion, skipped step, out of sequence, inaction, completion."""
        logger = EventLogger(log_dir=self.test_dir, custom_filename="events_test.jsonl")

        logger.log_event(0.1, 1, "EXPERIMENT_STARTED", "S01", 1, "OPEN_WHITE_BOX", "OPEN_WHITE_BOX", "START", 1.0, "Started")
        logger.log_step_completed(0.3, 3, "S01", 1, "OPEN_WHITE_BOX", "S02", 2, "RETRIEVE_RED_BOX", 0.95, "Completed S01")
        logger.log_event(1.0, 10, "SKIPPED_STEP", "S08", 8, "PICK_SPRAY", "SPRAY_PLANT", "ERROR", 0.88, "Skipped step S08", force=True)
        logger.log_event(2.0, 20, "OUT_OF_SEQUENCE", "S02", 2, "RETRIEVE_RED_BOX", "RETRIEVE_YELLOW_BOX", "ERROR", 0.85, "Out of sequence", force=True)
        logger.log_event(30.0, 300, "INACTION", "S03", 3, "RETRIEVE_YELLOW_BOX", "UNCERTAIN", "WAITING", 0.20, "Inactivity alert", force=True)
        logger.log_experiment_completed(50.0, 500, "S15", 15, "CLOSE_WHITE_BOX", 1.0, "Done")

        logger.close()

        with open(os.path.join(self.test_dir, "events_test.jsonl"), 'r', encoding='utf-8') as f:
            events = [json.loads(line)["event"] for line in f if line.strip()]

        self.assertIn("EXPERIMENT_STARTED", events)
        self.assertIn("STEP_COMPLETED", events)
        self.assertIn("SKIPPED_STEP", events)
        self.assertIn("OUT_OF_SEQUENCE", events)
        self.assertIn("INACTION", events)
        self.assertIn("EXPERIMENT_COMPLETED", events)

    def test_duplicate_suppression(self):
        """Duplicate events are suppressed and repeated errors do not create log spam."""
        logger = EventLogger(log_dir=self.test_dir, custom_filename="dedup_test.jsonl")

        logged_records = []
        for i in range(100):
            rec = logger.log_event(0.1 + (i * 0.01), i + 1, "SKIPPED_STEP", "S08", 8, "PICK_SPRAY", "SPRAY_PLANT", "ERROR", 0.88, "Skipped")
            if rec:
                logged_records.append(rec)

        logger.close()
        self.assertEqual(len(logged_records), 1)


if __name__ == "__main__":
    unittest.main()
