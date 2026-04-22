from __future__ import annotations

import time
import unittest

from agent.budget import BudgetExceededError, BudgetTracker


class BudgetTrackerTests(unittest.TestCase):
    def test_budget_remaining_ms_decreases_over_time(self) -> None:
        tracker = BudgetTracker(wall_clock_s=10.0, max_prompt_tokens=2000)
        r1 = tracker.remaining_ms()
        time.sleep(0.01)
        r2 = tracker.remaining_ms()
        self.assertLess(r2, r1)
        self.assertGreaterEqual(r2, 0)

    def test_budget_pre_generation_raises_when_wall_clock_exceeded(self) -> None:
        tracker = BudgetTracker(wall_clock_s=0.001, max_prompt_tokens=2000)
        time.sleep(0.01)
        with self.assertRaises(BudgetExceededError) as cm:
            tracker.check_pre_generation("retrieval_complete")
        self.assertEqual(cm.exception.kind, "wall_clock")
        self.assertEqual(cm.exception.label, "retrieval_complete")
        self.assertEqual(cm.exception.code, "BUDGET_EXCEEDED")

    def test_budget_during_generation_returns_degrade_flag_and_does_not_raise(self) -> None:
        tracker = BudgetTracker(wall_clock_s=0.001, max_prompt_tokens=2000)
        time.sleep(0.01)
        result = tracker.check_during_generation()
        self.assertIs(result, True)
        self.assertIs(tracker.degrade_flag(), True)
        tracker.check_during_generation()
        self.assertIs(tracker.snapshot()["degrade_during_generation"], True)

    def test_budget_record_prompt_tokens_raises_when_exceeded(self) -> None:
        tracker = BudgetTracker(wall_clock_s=10.0, max_prompt_tokens=100)
        tracker.record_prompt_tokens(40)
        tracker.record_prompt_tokens(50)
        with self.assertRaises(BudgetExceededError) as cm:
            tracker.record_prompt_tokens(20)
        self.assertEqual(cm.exception.kind, "tokens")
        # Failed call must NOT mutate state.
        self.assertEqual(tracker.prompt_tokens_used(), 90)

    def test_budget_invalid_constructor_args_raise(self) -> None:
        with self.assertRaises(ValueError):
            BudgetTracker(wall_clock_s=0, max_prompt_tokens=100)
        with self.assertRaises(ValueError):
            BudgetTracker(wall_clock_s=10, max_prompt_tokens=0)


if __name__ == "__main__":
    unittest.main()
