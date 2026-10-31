"""Integration test suite for LLM Router Admin & Dynamic Features."""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from app.main import app
from app.core.state_manager import state_manager


async def run_tests():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Dashboard Homepage
        r = await client.get("/")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        assert "LLM Router" in r.text
        assert "Models &amp; Toggles" in r.text or "Models & Toggles" in r.text
        assert "Routing Hierarchy" in r.text
        print("[PASS] Dashboard rendered successfully")

        # 2. Admin Models List
        r = await client.get("/api/admin/models")
        assert r.status_code == 200
        data = r.json()
        assert "providers" in data
        assert "antigravity" in data["providers"]
        assert "gemini" in data["providers"]
        assert "cerebras" in data["providers"]
        assert "groq" in data["providers"]
        print(f"[PASS] Admin models: {len(data['providers'])} providers discovered")

        # 3. Model Toggle
        r = await client.post(
            "/api/admin/models/toggle",
            json={"provider": "groq", "model": "openai/gpt-oss-20b", "disabled": True},
        )
        assert r.status_code == 200
        assert r.json()["disabled"] is True
        assert state_manager.is_model_disabled("groq", "openai/gpt-oss-20b") is True
        print("[PASS] Model successfully disabled")

        r = await client.post(
            "/api/admin/models/toggle",
            json={"provider": "groq", "model": "openai/gpt-oss-20b", "disabled": False},
        )
        assert r.status_code == 200
        assert r.json()["disabled"] is False
        assert state_manager.is_model_disabled("groq", "openai/gpt-oss-20b") is False
        print("[PASS] Model successfully re-enabled")

        # 4. Strategy Management
        r = await client.get("/api/admin/strategy")
        assert r.status_code == 200
        st_data = r.json()
        assert "presets" in st_data
        assert "max_speed" in st_data["presets"]
        assert "best_quality" in st_data["presets"]
        assert "max_quota" in st_data["presets"]
        print("[PASS] Routing strategy presets retrieved")

        # Set to max_speed
        r = await client.post("/api/admin/strategy/set", json={"strategy": "max_speed"})
        assert r.status_code == 200
        assert r.json()["active_strategy"] == "max_speed"
        assert state_manager.get_active_strategy() == "max_speed"
        print("[PASS] Strategy successfully changed to max_speed")

        # Custom priority hierarchy
        custom_priority = [
            ["cerebras", "gpt-oss-120b"],
            ["gemini", "gemini-3.6-flash"],
            ["groq", "openai/gpt-oss-120b"],
        ]
        r = await client.post(
            "/api/admin/strategy/custom-priority",
            json={"priority": custom_priority},
        )
        assert r.status_code == 200
        assert state_manager.get_custom_priority() == custom_priority
        print("[PASS] Custom priority hierarchy saved")

        # Reset to default
        r = await client.post("/api/admin/strategy/set", json={"strategy": "default"})
        assert r.status_code == 200
        print("[PASS] Strategy reset to default")

        # 5. Mega Pool Alias
        r = await client.get("/v1/models")
        assert r.status_code == 200
        models_data = r.json()
        model_ids = [m["id"] for m in models_data.get("data", [])]
        assert "mega-pool" in model_ids
        print("[PASS] mega-pool virtual model alias is registered")

        # 6. Request History & IST Timestamps
        from app.core.history_manager import history_manager
        test_rec = history_manager.record_request(
            endpoint="/v1/chat/completions",
            protocol="openai",
            requested_model="mega-pool",
            resolved_provider="cerebras",
            resolved_model="llama-3.3-70b",
            status="fulfilled",
            http_status=200,
            prompt_tokens=10,
            completion_tokens=25,
            latency_ms=250.0,
            tier="CODING",
        )
        assert test_rec["timestamp"].endswith(" AM IST") or test_rec["timestamp"].endswith(" PM IST"), f"Expected 12-hour AM/PM IST timestamp, got: {test_rec['timestamp']}"
        r = await client.get("/api/admin/history?limit=10")
        assert r.status_code == 200
        hist_data = r.json()
        assert len(hist_data["records"]) > 0
        latest = hist_data["records"][0]
        assert latest["timestamp"].endswith(" AM IST") or latest["timestamp"].endswith(" PM IST"), f"API record timestamp must be 12-hour AM/PM IST, got: {latest['timestamp']}"
        print(f"[PASS] Request history returns valid 12-hour AM/PM IST timestamps: {latest['timestamp']}")

    print("\n[SUCCESS] ALL ADMIN & DYNAMIC ROUTING TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(run_tests())
