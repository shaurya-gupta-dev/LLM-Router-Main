"""Unit & integration tests for Dynamic Waterfall Cascades and State-aware routing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.router import router_engine
from app.core.state_manager import state_manager


def test_dynamic_waterfall():
    print("Testing dynamic waterfall cascades...")

    # 1. Base chain resolution for mega-pool
    chain = router_engine.resolve_fallback_chain("mega-pool")
    assert len(chain) >= 8, f"Expected at least 8 models in mega-pool, got {len(chain)}"
    assert ("gemini", "gemini-3.8-flash") in chain
    assert ("cerebras", "gpt-oss-120b") in chain
    print(f"[PASS] mega-pool resolved {len(chain)} candidates")

    # 2. Model disabling filters out disabled model
    state_manager.toggle_model("gemini", "gemini-3.8-flash", disabled=True)
    chain_after_disable = router_engine.resolve_fallback_chain("mega-pool")
    assert ("gemini", "gemini-3.8-flash") not in chain_after_disable
    print("[PASS] Disabled model filtered out from cascade chain")

    # Re-enable
    state_manager.toggle_model("gemini", "gemini-3.8-flash", disabled=False)
    chain_after_reenable = router_engine.resolve_fallback_chain("mega-pool")
    assert ("gemini", "gemini-3.8-flash") in chain_after_reenable
    print("[PASS] Re-enabled model restored to cascade chain")

    # 3. Strategy: max_speed puts fastest providers first
    state_manager.set_active_strategy("max_speed")
    chain_speed = router_engine.resolve_fallback_chain("claude-3-7-sonnet-latest")
    assert chain_speed[0] == ("cerebras", "gpt-oss-120b")
    print(f"[PASS] max_speed strategy prioritized: {chain_speed[0]}")

    # 4. Strategy: best_quality
    state_manager.set_active_strategy("best_quality")
    chain_quality = router_engine.resolve_fallback_chain("claude-3-7-sonnet-latest")
    assert chain_quality[0] == ("gemini", "gemini-3.8-flash")
    print(f"[PASS] best_quality strategy prioritized: {chain_quality[0]}")

    # 5. Strategy: custom
    state_manager.set_custom_priority([
        ["sambanova", "gpt-oss-120b"],
        ["groq", "openai/gpt-oss-120b"],
    ])
    state_manager.set_active_strategy("custom")
    chain_custom = router_engine.resolve_fallback_chain("claude-3-7-sonnet-latest")
    assert chain_custom[0] == ("sambanova", "gpt-oss-120b")
    assert chain_custom[1] == ("groq", "openai/gpt-oss-120b")
    print(f"[PASS] custom hierarchy prioritized: {chain_custom[:2]}")

    # Reset to default
    state_manager.set_active_strategy("default")
    print("[PASS] Successfully reset strategy to default")

    print("\n[SUCCESS] ALL DYNAMIC WATERFALL & FILTERING TESTS PASSED!")


if __name__ == "__main__":
    test_dynamic_waterfall()
