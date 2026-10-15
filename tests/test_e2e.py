import asyncio
import os
import httpx
import json
from dotenv import load_dotenv

load_dotenv()

ROUTER_URL = "http://localhost:8080"
MASTER_KEY = os.getenv("MASTER_KEY", "llm-router-secret-key-12345")

async def test_health():
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{ROUTER_URL}/api/health")
        print(f"[HEALTH] Status: {r.status_code}")
        print(f"[HEALTH] Response: {r.json()}")
        return r.status_code == 200

async def test_config():
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{ROUTER_URL}/api/config")
        print(f"\n[CONFIG] Status: {r.status_code}")
        data = r.json()
        print(f"[CONFIG] Master key: {data.get('master_key')}")
        print(f"[CONFIG] Providers: {data.get('providers_configured')}")
        print(f"[CONFIG] Models: {data.get('models_configured')}")
        return r.status_code == 200

async def test_models():
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{ROUTER_URL}/v1/models")
        print(f"\n[MODELS] Status: {r.status_code}")
        data = r.json()
        models = data.get("data", [])
        print(f"[MODELS] {len(models)} models listed:")
        for m in models[:5]:
            print(f"  - {m.get('id')}")
        return r.status_code == 200

async def test_auth_fail():
    """Test that wrong key gets rejected."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(
            f"{ROUTER_URL}/v1/messages",
            headers={"Content-Type": "application/json", "x-api-key": "wrong-key"},
            json={"model": "claude-3-7-sonnet-latest", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}
        )
        print(f"\n[AUTH FAIL TEST] Status: {r.status_code} (expected 401)")
        return r.status_code == 401

async def test_anthropic_messages(stream=False):
    """Test the /v1/messages endpoint (Anthropic format)."""
    print(f"\n[/v1/messages stream={stream}] Testing...")
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(
                f"{ROUTER_URL}/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": MASTER_KEY,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-3-7-sonnet-latest",
                    "messages": [{"role": "user", "content": "Say 'LLM Router is working!' and nothing else."}],
                    "max_tokens": 256,
                    "stream": stream,
                }
            )
            
            print(f"[/v1/messages stream={stream}] Status: {r.status_code}")
            provider = r.headers.get("X-Routed-Provider", "unknown")
            model = r.headers.get("X-Routed-Model", "unknown")
            print(f"[/v1/messages stream={stream}] Routed via: {provider}/{model}")
            
            if r.status_code == 200:
                if stream:
                    text_parts = []
                    for line in r.text.split("\n"):
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                ev = json.loads(line[6:])
                                if ev.get("type") == "content_block_delta":
                                    delta = ev.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        text_parts.append(delta.get("text", ""))
                            except Exception:
                                pass
                    text = "".join(text_parts)
                    print(f"[/v1/messages stream=True] Streamed text: {text[:100]}")
                else:
                    data = r.json()
                    content = (data.get("content") or [{}])[0].get("text", "")
                    stop_reason = data.get("stop_reason")
                    print(f"[/v1/messages stream=False] Content: {content[:100]}")
                    print(f"[/v1/messages stream=False] Stop reason: {stop_reason}")
                return True
            else:
                print(f"[/v1/messages stream={stream}] Error: {r.text[:300]}")
                return False
    except Exception as e:
        print(f"[/v1/messages stream={stream}] Exception: {e}")
        return False

async def test_openai_chat():
    """Test the /v1/chat/completions endpoint (OpenAI format)."""
    print(f"\n[/v1/chat/completions] Testing...")
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(
                f"{ROUTER_URL}/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {MASTER_KEY}",
                },
                json={
                    "model": "claude-3-7-sonnet-latest",
                    "messages": [{"role": "user", "content": "Say 'OpenAI format works!' and nothing else."}],
                    "max_tokens": 256,
                }
            )
            print(f"[/v1/chat/completions] Status: {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"[/v1/chat/completions] Content: {content[:100]}")
                return True
            else:
                print(f"[/v1/chat/completions] Error: {r.text[:300]}")
                return False
    except Exception as e:
        print(f"[/v1/chat/completions] Exception: {e}")
        return False

async def main():
    print("=" * 65)
    print("  LLM ROUTER - FULL END-TO-END TEST SUITE")
    print("=" * 65)
    
    results = {}
    
    results["health"] = await test_health()
    results["config"] = await test_config()
    results["models"] = await test_models()
    results["auth_fail"] = await test_auth_fail()
    results["messages_non_stream"] = await test_anthropic_messages(stream=False)
    results["messages_stream"] = await test_anthropic_messages(stream=True)
    results["openai_chat"] = await test_openai_chat()
    
    print("\n" + "=" * 65)
    print("  TEST RESULTS SUMMARY")
    print("=" * 65)
    all_passed = True
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False
    
    print("=" * 65)
    if all_passed:
        print("  ALL TESTS PASSED - LLM ROUTER IS FULLY OPERATIONAL!")
    else:
        print("  SOME TESTS FAILED - CHECK OUTPUT ABOVE")
    print("=" * 65)

asyncio.run(main())
