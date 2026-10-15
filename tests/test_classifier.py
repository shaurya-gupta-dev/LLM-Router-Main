"""Unit tests for the Intelligent Hybrid Query Classifier."""

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.classifier import QueryClassifier


class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = QueryClassifier()

    def test_casual_heuristic(self):
        casual_prompts = [
            "Hi there!",
            "Hello, how are you?",
            "Tell me a joke",
            "Thank you so much",
            "Good morning assistant",
            "Please summarize this article for me",
        ]
        for p in casual_prompts:
            tier = self.classifier.classify_heuristic(p)
            self.assertEqual(tier, "CASUAL", f"Failed to classify '{p}' as CASUAL (got {tier})")

    def test_coding_heuristic(self):
        coding_prompts = [
            "Write a Python function to sort a list using quicksort",
            "def reverse_string(s: str) -> str:",
            "Fix this error:\nTraceback (most recent call last):\n  File 'app.py', line 10\nTypeError: NoneType",
            "const [state, setState] = useState(0);",
            "SELECT id, name FROM users WHERE active = 1;",
            "Refactor this async connection pool logic",
            "How do I use docker-compose to spin up postgres?",
        ]
        for p in coding_prompts:
            tier = self.classifier.classify_heuristic(p)
            self.assertEqual(tier, "CODING", f"Failed to classify '{p}' as CODING (got {tier})")

    def test_reasoning_heuristic(self):
        reasoning_prompts = [
            "Design a distributed system architecture for processing 10 million transactions per second with low latency tradeoffs",
            "Explain the mathematical proof for Fermat's Last Theorem using step by step reasoning",
            "Analyze the concurrency and deadlock risks in this distributed consensus protocol",
        ]
        for p in reasoning_prompts:
            tier = self.classifier.classify_heuristic(p)
            self.assertEqual(tier, "REASONING", f"Failed to classify '{p}' as REASONING (got {tier})")

    def test_async_classify_with_heuristic(self):
        async def run_test():
            messages = [{"role": "user", "content": "Hello! How can you help me today?"}]
            tier, method, ms = await self.classifier.classify(messages)
            self.assertEqual(tier, "CASUAL")
            self.assertEqual(method, "heuristic")
            self.assertLess(ms, 10.0)

        asyncio.run(run_test())

    def test_async_classify_with_micro_llm_mock(self):
        async def run_test():
            # Ambiguous prompt that doesn't trigger instant regex
            messages = [{"role": "user", "content": "Should we migrate from monolith to microservices?"}]
            
            async def mock_micro(prompt: str) -> str:
                return "R"  # Reasoning

            tier, method, ms = await self.classifier.classify(messages, micro_infer_func=mock_micro)
            self.assertEqual(tier, "REASONING")
            self.assertEqual(method, "micro_llm")

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
