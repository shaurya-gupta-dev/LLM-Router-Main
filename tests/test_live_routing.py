"""Live end-to-end test with real keys and model fallbacks."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from app.core.config import config
from app.main import app

async def run_live_test():
    print("\n--> 1. Testing Live Anthropic /v1/messages (Non-Streaming)...")
    transport = httpx.ASGITransport(app=app)
    master_key = config.settings.master_key

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/messages",
            headers={
                "x-api-key": master_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-3-7-sonnet-latest",
                "messages": [{"role": "user", "content": "Respond with 2 words: Routing Works"}],
                "max_tokens": 256,
                "stream": False,
            },
            timeout=30.0,
        )
        print("Status Code:", resp.status_code)
        if resp.status_code == 200:
            data = resp.json()
            prov = resp.headers.get("X-Routed-Provider")
            model = resp.headers.get("X-Routed-Model")
            text = (data.get("content") or [{}])[0].get("text", "")
            print(f"[PASS] Successfully routed non-streaming to: {prov} ({model}) -> \"{text}\"")
        else:
            print("[FAIL] Error:", resp.text)

        print("\n--> 2. Testing Live Anthropic /v1/messages (Streaming SSE)...")
        stream_resp = await client.post(
            "/v1/messages",
            headers={
                "x-api-key": master_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Count from 1 to 5"}],
                "max_tokens": 256,
                "stream": True,
            },
            timeout=30.0,
        )
        print("Streaming Status Code:", stream_resp.status_code)
        if stream_resp.status_code == 200:
            print("Received SSE Stream Events:")
            accumulated_text = ""
            for line in stream_resp.text.split("\n"):
                if line.startswith("data: "):
                    try:
                        ev = json.loads(line[6:])
                        if ev.get("type") == "content_block_delta":
                            delta = ev.get("delta", {}).get("text", "")
                            accumulated_text += delta
                    except Exception:
                        pass
            print(f"[PASS] Streamed Text: \"{accumulated_text.strip()}\"")
        else:
            print("[FAIL] Streaming error:", stream_resp.text)

        print("\n--> 3. Testing OpenAI /v1/chat/completions with gpt-4o...")
        oa_resp = await client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {master_key}",
            },
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Respond with 1 word: Verified"}],
                "max_tokens": 20,
            },
            timeout=30.0,
        )
        print("OpenAI Status Code:", oa_resp.status_code)
        if oa_resp.status_code == 200:
            oa_data = oa_resp.json()
            oa_text = oa_data["choices"][0]["message"]["content"]
            prov = oa_resp.headers.get("X-Routed-Provider")
            model = oa_resp.headers.get("X-Routed-Model")
            print(f"[PASS] Routed OpenAI to {prov} ({model}): \"{oa_text.strip()}\"")
        else:
            print("[FAIL] OpenAI error:", oa_resp.text)

if __name__ == "__main__":
    asyncio.run(run_live_test())
