import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.rate_budget import ProviderBudgetLimits, ProviderBudgetTracker, RateBudgetManager


class TestRateBudget(unittest.TestCase):
    def test_sliding_window_tracking(self):
        tracker = ProviderBudgetTracker(
            "test_prov",
            ProviderBudgetLimits(rpm_limit=10, rpd_limit=100, soft_threshold_pct=0.7),
        )
        now = time.time()
        for _ in range(5):
            tracker.record_request(now)

        rpm, rpd = tracker.current_usage(now)
        self.assertEqual(rpm, 5)
        self.assertEqual(rpd, 5)

        # 5/10 is under 70% (7), so available
        avail, reason = tracker.is_available(now)
        self.assertTrue(avail)
        self.assertEqual(reason, "OK")

    def test_soft_threshold_preemptive_cooldown(self):
        tracker = ProviderBudgetTracker(
            "test_prov",
            ProviderBudgetLimits(rpm_limit=10, rpd_limit=100, soft_threshold_pct=0.7),
        )
        now = time.time()
        # Add 7 requests (hits 70% of 10)
        for _ in range(7):
            tracker.record_request(now)

        # Standard check should be soft-throttled
        avail, reason = tracker.is_available(now, allow_soft_throttled=False)
        self.assertFalse(avail)
        self.assertIn("Soft budget reached", reason)

        # If allow_soft_throttled is True (fallback pass), it should allow using remaining headroom
        avail_fallback, _ = tracker.is_available(now, allow_soft_throttled=True)
        self.assertTrue(avail_fallback)

    def test_hard_limit_rejection(self):
        tracker = ProviderBudgetTracker(
            "test_prov",
            ProviderBudgetLimits(rpm_limit=10, rpd_limit=100, soft_threshold_pct=0.7),
        )
        now = time.time()
        for _ in range(10):
            tracker.record_request(now)

        # At 10/10, even fallback pass should reject
        avail, reason = tracker.is_available(now, allow_soft_throttled=True)
        self.assertFalse(avail)
        self.assertIn("RPM limit reached", reason)

    def test_model_specific_budgets(self):
        mgr = RateBudgetManager()
        # Verify calibrated limits are registered
        status = mgr.get_status()
        self.assertIn("gemini:gemini-3.5-flash-lite", status)
        self.assertIn("gemini:gemini-3.1-flash-lite", status)
        self.assertIn("gemini:gemma-4-31b-it", status)
        self.assertIn("groq:openai/gpt-oss-120b", status)

        self.assertEqual(status["gemini:gemini-3.5-flash-lite"]["rpd_limit"], 500)
        self.assertEqual(status["gemini:gemma-4-31b-it"]["rpd_limit"], 14000)
        self.assertEqual(status["groq:openai/gpt-oss-120b"]["rpd_limit"], 1000)

        # Check availability with model parameter
        avail, reason = mgr.check_availability("gemini", model="gemini-3.5-flash-lite")
        self.assertTrue(avail)
        self.assertEqual(reason, "OK")


if __name__ == "__main__":
    unittest.main()
