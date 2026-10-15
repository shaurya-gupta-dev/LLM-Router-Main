import os
import httpx
import sys
from dotenv import load_dotenv

load_dotenv()

URL = "http://localhost:8080"
KEY = os.getenv("MASTER_KEY", "llm-router-secret-key-12345")
models = [
    "claude-3-7-sonnet-latest",
    "claude-3-5-sonnet-20241022",
    "claude-opus-4-5",
    "claude-3-5-haiku-20241022",
    "gpt-4o",
    "gpt-4o-mini",
    "deepseek-chat",
]

print(f"| {'Alias':<28} | {'Protocol':<10} | {'Status':<6} | {'Routed Via':<36} |")
print("|" + "-"*30 + "|" + "-"*12 + "|" + "-"*8 + "|" + "-"*38 + "|")

all_ok = True
for m in models:
    # Anthropic
    try:
        r = httpx.post(
            f"{URL}/v1/messages",
            headers={"x-api-key": KEY, "Content-Type": "application/json"},
            json={"model": m, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5},
            timeout=30.0,
        )
        routed = f"{r.headers.get('X-Routed-Provider', '?')}/{r.headers.get('X-Routed-Model', '?')}"
        status_str = f"{r.status_code}"
        if r.status_code != 200:
            all_ok = False
        print(f"| {m:<28} | Anthropic  | {status_str:<6} | {routed:<36} |")
    except Exception as e:
        all_ok = False
        print(f"| {m:<28} | Anthropic  | ERR    | {str(e)[:36]:<36} |")

    # OpenAI
    try:
        r = httpx.post(
            f"{URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
            json={"model": m, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5},
            timeout=30.0,
        )
        routed = f"{r.headers.get('X-Routed-Provider', '?')}/{r.headers.get('X-Routed-Model', '?')}"
        status_str = f"{r.status_code}"
        if r.status_code != 200:
            all_ok = False
        print(f"| {m:<28} | OpenAI     | {status_str:<6} | {routed:<36} |")
    except Exception as e:
        all_ok = False
        print(f"| {m:<28} | OpenAI     | ERR    | {str(e)[:36]:<36} |")

print("\nMatrix status:", "ALL PASSED" if all_ok else "FAILURES DETECTED")
sys.exit(0 if all_ok else 1)
