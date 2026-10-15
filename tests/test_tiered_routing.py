"""Integration tests for Intelligent Tiered Routing & Custom Diagnostics."""

import asyncio
import os
import sys
import unittest
from pathlib import Path
import httpx
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROUTER_URL = "http://localhost:8080"
MASTER_KEY = os.getenv("MASTER_KEY", "llm-router-secret-key-12345")


class TestTieredRouting(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(timeout=60.0)

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_casual_query_tiering(self):
        """Casual greeting/joke should be classified as CASUAL."""
        r = await self.client.post(
            f"{ROUTER_URL}/v1/messages",
            headers={"x-api-key": MASTER_KEY, "Content-Type": "application/json"},
            json={
                "model": "claude-3-7-sonnet-latest",
                "messages": [{"role": "user", "content": "Hi there! Tell me a one line joke."}],
                "max_tokens": 30,
            },
        )
        self.assertEqual(r.status_code, 200, f"Error: {r.text}")
        tier = r.headers.get("X-Router-Tier")
        prov = r.headers.get("X-Routed-Provider")
        self.assertEqual(tier, "CASUAL")
        self.assertIn(prov, ["openrouter", "gemini", "groq", "cerebras", "cohere"])

    async def test_coding_query_tiering(self):
        """Code prompt should be classified as CODING."""
        r = await self.client.post(
            f"{ROUTER_URL}/v1/messages",
            headers={"x-api-key": MASTER_KEY, "Content-Type": "application/json"},
            json={
                "model": "claude-3-7-sonnet-latest",
                "messages": [{"role": "user", "content": "def bubble_sort(arr: list) -> list:\n    # implement bubble sort"}],
                "max_tokens": 40,
            },
        )
        self.assertEqual(r.status_code, 200, f"Error: {r.text}")
        tier = r.headers.get("X-Router-Tier")
        self.assertEqual(tier, "CODING")

    async def test_reasoning_query_tiering(self):
        """System architecture prompt should be classified as REASONING."""
        r = await self.client.post(
            f"{ROUTER_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {MASTER_KEY}", "Content-Type": "application/json"},
            json={
                "model": "claude-3-7-sonnet-latest",
                "messages": [{"role": "user", "content": "Analyze the system design tradeoffs of Raft vs Paxos consensus in distributed systems"}],
                "max_tokens": 40,
            },
        )
        self.assertEqual(r.status_code, 200, f"Error: {r.text}")
        tier = r.headers.get("X-Router-Tier")
        self.assertEqual(tier, "REASONING")

    async def test_preemptive_rate_budget_headers(self):
        """Telemetry headers must be present on responses."""
        r = await self.client.get(f"{ROUTER_URL}/api/stats")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("rate_budget", data)
        budget = data["rate_budget"]
        self.assertIn("openrouter", budget)
        self.assertIn("gemini", budget)


if __name__ == "__main__":
    unittest.main()
