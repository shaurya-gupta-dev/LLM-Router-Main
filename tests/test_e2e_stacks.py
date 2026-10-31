import asyncio
import json
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.environ.get("ROUTER_URL", "http://127.0.0.1:8080")
MASTER_KEY = os.environ.get("ROUTER_MASTER_KEY", os.environ.get("MASTER_KEY", "llm-router-secret-key-12345"))

HEADERS_OPENAI = {
    "Authorization": f"Bearer {MASTER_KEY}",
    "Content-Type": "application/json"
}
HEADERS_ANTHROPIC = {
    "x-api-key": MASTER_KEY,
    "anthropic-version": "2023-06-01",
    "Content-Type": "application/json"
}

async def test_openai_chat_completions():
    print("\n--- 1. Testing OpenAI /v1/chat/completions (hermes-agent) ---")
    payload = {
        "model": "hermes-agent",
        "messages": [
            {"role": "user", "content": "Compute 23 * 47 and reply concisely with only the number."}
        ],
        "temperature": 0.1,
        "max_tokens": 50
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{BASE_URL}/v1/chat/completions", headers=HEADERS_OPENAI, json=payload)
        print(f"Status: {resp.status_code}")
        assert resp.status_code == 200, f"Error: {resp.text}"
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        resolved = data.get("router_resolved", {})
        print(f"Response: {content.strip()}")
        print(f"Resolved Provider/Model: {resolved.get('provider')}:{resolved.get('model')}")

async def test_anthropic_messages_tool_call():
    print("\n--- 2. Testing Anthropic /v1/messages with Tool Calling (pro-coding) ---")
    payload = {
        "model": "pro-coding",
        "messages": [
            {"role": "user", "content": "What is the weather in Tokyo right now?"}
        ],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get current temperature for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }
            }
        ],
        "max_tokens": 128
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{BASE_URL}/v1/messages", headers=HEADERS_ANTHROPIC, json=payload)
        print(f"Status: {resp.status_code}")
        assert resp.status_code == 200, f"Error: {resp.text}"
        data = resp.json()
        print(f"Content Blocks: {json.dumps(data.get('content', []), indent=2)}")

async def test_openai_streaming():
    print("\n--- 3. Testing OpenAI Streaming SSE (ultra-fast-planning) ---")
    payload = {
        "model": "ultra-fast-planning",
        "messages": [
            {"role": "user", "content": "Count from 1 to 5 separated by commas."}
        ],
        "stream": True,
        "max_tokens": 50
    }
    chunks = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream("POST", f"{BASE_URL}/v1/chat/completions", headers=HEADERS_OPENAI, json=payload) as resp:
            print(f"Stream Status: {resp.status_code}")
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk_data = json.loads(line[6:])
                    delta = chunk_data["choices"][0].get("delta", {}).get("content", "")
                    chunks.append(delta)
    streamed_text = "".join(chunks)
    print(f"Streamed Output: {streamed_text.strip()}")

async def test_telemetry_live_tracking():
    print("\n--- 4. Checking /api/admin/usage Telemetry ---")
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{BASE_URL}/api/admin/usage")
        assert resp.status_code == 200
        data = resp.json()
        summary = data["summary"]
        print(f"Total Requests (24h): {summary.get('today_requests', 0)}")
        print(f"Active Keys: {summary.get('active_keys_count', 0)} / {summary.get('total_pooled_keys', 0)}")
        print(f"Top Models Count: {len(data.get('top_models', []))}")
        for m in data.get("top_models", [])[:3]:
            print(f"  * {m['model_id']} ({m['provider_id']}): {m['total_requests']} reqs, {m['total_tokens']} tokens")

async def main():
    await test_openai_chat_completions()
    await test_anthropic_messages_tool_call()
    await test_openai_streaming()
    await test_telemetry_live_tracking()
    print("\n=== All End-to-End Tests Succeeded! ===")

if __name__ == "__main__":
    asyncio.run(main())
