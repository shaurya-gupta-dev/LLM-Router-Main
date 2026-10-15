"""Integration test suite for LLM Router."""

import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from app.main import app

async def run_tests():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Dashboard
        r = await client.get("/")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        assert "LLM Router" in r.text

        # 2. Models
        r = await client.get("/v1/models")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        data = r.json()
        ids = [m["id"] for m in data.get("data", [])]
        assert "claude-3-7-sonnet-latest" in ids
        assert "claude-3-5-sonnet-20241022" in ids
        assert "claude-3-5-haiku-20241022" in ids
        assert "gpt-4o" in ids

        # 3. Stats & Key Pools
        r = await client.get("/api/stats")
        assert r.status_code == 200
        stats = r.json()
        pools = stats.get("key_pools", {})
        assert len(pools) >= 10

        # 4. Auth
        r = await client.post("/v1/messages", json={"model": "claude-3-5-sonnet-20241022", "messages": []})
        assert r.status_code == 401

    print("[SUCCESS] All LLM Router tests passed.")

if __name__ == "__main__":
    asyncio.run(run_tests())
