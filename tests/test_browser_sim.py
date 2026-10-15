"""Test browser playground simulation (both non-streaming and streaming)."""
import asyncio
import httpx
import json

async def test_browser_playground_simulation():
    async with httpx.AsyncClient(timeout=30) as client:
        # Step 1: fetchConfig (as done on page load)
        cfg_resp = await client.get("http://localhost:8080/api/config")
        assert cfg_resp.status_code == 200
        cfg = cfg_resp.json()
        master_key = cfg["master_key"]
        print(f"[BROWSER SIM] Loaded master_key from /api/config: {master_key}")

        # Step 2: fetchChains (as done on page load)
        chains_resp = await client.get("http://localhost:8080/api/chains")
        assert chains_resp.status_code == 200
        chains = chains_resp.json()
        print(f"[BROWSER SIM] Loaded {len(chains)} fallback chains")

        # Step 3: Playground Non-streaming request (simulating 'Send Request' button unchecked stream)
        play_payload = {
            "model": "claude-3-7-sonnet-latest",
            "system": "You are a helpful coding assistant.",
            "messages": [{"role": "user", "content": "Say hello in 3 words"}],
            "stream": False,
            "max_tokens": 256,
        }
        r_non_stream = await client.post(
            "http://localhost:8080/v1/messages",
            headers={"Content-Type": "application/json", "x-api-key": master_key},
            json=play_payload,
        )
        print(f"[BROWSER SIM] Non-streaming status: {r_non_stream.status_code}")
        assert r_non_stream.status_code == 200
        data = r_non_stream.json()
        print(f"[BROWSER SIM] Non-streaming response text: {data['content'][0]['text'][:80]}")
        print(f"[BROWSER SIM] Routed Provider: {r_non_stream.headers.get('X-Routed-Provider')}")

        # Step 4: Playground Streaming request (simulating 'Send Request' button checked stream)
        play_payload["stream"] = True
        r_stream = await client.post(
            "http://localhost:8080/v1/messages",
            headers={"Content-Type": "application/json", "x-api-key": master_key},
            json=play_payload,
        )
        print(f"[BROWSER SIM] Streaming status: {r_stream.status_code}")
        assert r_stream.status_code == 200
        chunks = []
        for line in r_stream.text.split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    ev = json.loads(line[6:])
                    if ev.get("type") == "content_block_delta":
                        chunks.append(ev.get("delta", {}).get("text", ""))
                except Exception:
                    pass
        streamed_output = "".join(chunks)
        print(f"[BROWSER SIM] Streaming output received: {streamed_output[:80]}")
        assert len(streamed_output) > 0

    print("\n[BROWSER SIM SUCCESS] Dashboard and Playground workflow 100% verified!")

asyncio.run(test_browser_playground_simulation())
