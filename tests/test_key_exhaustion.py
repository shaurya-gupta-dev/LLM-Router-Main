"""Unit and integration test suite for multi-key discovery and per-model key exhaustion."""

import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import config, ProviderDefinition, FreeTierConfig
from app.core.key_pool import KeyPoolManager, ProviderKeyPool, KeyState
from app.core.router import router_engine
from app.providers.base import UnifiedChatResult, RateLimitException


class TestKeyExhaustion(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Fresh key manager for isolated test cases
        self.km = KeyPoolManager()

    def test_key_discovery_formats(self):
        """Verify delimiter parsing (comma, semicolon, newline) and numbered variables."""
        test_provider_id = "test_multi"
        config.providers[test_provider_id] = ProviderDefinition(
            name="Test Multi",
            type="openai_compat",
            base_url="https://api.test.com",
            env_keys=["TEST_MULTI_API_KEYS", "TEST_MULTI_API_KEY"],
            free_tier=FreeTierConfig(rpm=10, rpd=100),
        )

        # 1. Comma, semicolon, newline mixed in one variable
        os.environ["TEST_MULTI_API_KEYS"] = "key_alpha, key_beta; key_gamma\nkey_delta"
        keys = config.get_provider_keys(test_provider_id)
        self.assertEqual(keys, ["key_alpha", "key_beta", "key_gamma", "key_delta"])

        # 2. Numbered variables
        del os.environ["TEST_MULTI_API_KEYS"]
        os.environ["TEST_MULTI_API_KEY_1"] = "num_key_1"
        os.environ["TEST_MULTI_API_KEY_2"] = "num_key_2"
        os.environ["TEST_MULTI_API_KEY_3"] = "num_key_3"
        keys_num = config.get_provider_keys(test_provider_id)
        self.assertEqual(keys_num, ["num_key_1", "num_key_2", "num_key_3"])

        # Clean up
        for k in ["TEST_MULTI_API_KEY_1", "TEST_MULTI_API_KEY_2", "TEST_MULTI_API_KEY_3"]:
            os.environ.pop(k, None)
        config.providers.pop(test_provider_id, None)

    def test_pool_get_next_key_excluding(self):
        """Verify round-robin selection and in-request exclusion set."""
        pool = ProviderKeyPool("prov_a", ["k1", "k2", "k3"])

        # First request picks k1
        k_first = pool.get_next_key_excluding(set())
        self.assertIsNotNone(k_first)
        self.assertEqual(k_first.key, "k1")

        # In same request, k1 failed, exclude k1 -> picks k2
        k_second = pool.get_next_key_excluding({"k1"})
        self.assertIsNotNone(k_second)
        self.assertEqual(k_second.key, "k2")

        # In same request, k2 also failed, exclude {k1, k2} -> picks k3
        k_third = pool.get_next_key_excluding({"k1", "k2"})
        self.assertIsNotNone(k_third)
        self.assertEqual(k_third.key, "k3")

        # All keys excluded -> returns None
        k_none = pool.get_next_key_excluding({"k1", "k2", "k3"})
        self.assertIsNone(k_none)

    async def test_chat_exhausts_keys_before_model_failover(self):
        """
        Verify that when Key 1 fails on Model A with 429, Key 2 on the SAME Model A
        is attempted and succeeds before any fallback to Model B occurs.
        """
        from app.core.key_pool import key_manager

        # Set up test providers with multiple keys
        key_manager.register_provider("prov_primary", ["key_prim_1", "key_prim_2"])
        key_manager.register_provider("prov_backup", ["key_back_1"])

        mock_primary = AsyncMock()
        mock_backup = AsyncMock()

        # Call record tracking
        call_log = []

        async def primary_chat(model, messages, api_key, **kwargs):
            call_log.append((model, api_key))
            if api_key == "key_prim_1":
                raise RateLimitException("429 Too Many Requests on key 1", retry_after=30.0)
            return UnifiedChatResult(
                content="Success from primary key 2!",
                prompt_tokens=10,
                completion_tokens=5,
            )

        async def backup_chat(model, messages, api_key, **kwargs):
            call_log.append((model, api_key))
            return UnifiedChatResult(
                content="Success from backup!",
                prompt_tokens=10,
                completion_tokens=5,
            )

        mock_primary.chat_completion.side_effect = primary_chat
        mock_backup.chat_completion.side_effect = backup_chat

        router_engine.providers["prov_primary"] = mock_primary
        router_engine.providers["prov_backup"] = mock_backup

        with patch.object(
            router_engine,
            "resolve_fallback_chain",
            return_value=[("prov_primary", "model-primary"), ("prov_backup", "model-backup")],
        ):
            res, p_resolved, m_resolved, meta = await router_engine.execute_chat(
                requested_model="test-stack",
                messages=[{"role": "user", "content": "hello"}],
            )

            # Assert request succeeded on the PRIMARY model, NOT backup!
            self.assertEqual(p_resolved, "prov_primary")
            self.assertEqual(m_resolved, "model-primary")
            self.assertEqual(res.content, "Success from primary key 2!")

            # Assert call log shows key_prim_1 was tried first, then key_prim_2 on the SAME model
            self.assertEqual(len(call_log), 2)
            self.assertEqual(call_log[0], ("model-primary", "key_prim_1"))
            self.assertEqual(call_log[1], ("model-primary", "key_prim_2"))

            # Assert attempted candidates in metadata show key failover
            attempts = meta["candidates_attempted"]
            self.assertEqual(len(attempts), 2)
            self.assertEqual(attempts[0]["status"], "rate_limited")
            self.assertEqual(attempts[0]["key"], "...prim_1")
            self.assertEqual(attempts[1]["status"], "success")
            self.assertEqual(attempts[1]["key"], "...prim_2")

    async def test_chat_cascades_only_after_all_keys_exhausted(self):
        """
        Verify that only after ALL keys for Model A fail with 429 does the router
        cascade to Model B.
        """
        from app.core.key_pool import key_manager

        key_manager.register_provider("prov_primary", ["key_prim_1", "key_prim_2"])
        key_manager.register_provider("prov_backup", ["key_back_1"])

        mock_primary = AsyncMock()
        mock_backup = AsyncMock()

        call_log = []

        async def primary_chat(model, messages, api_key, **kwargs):
            call_log.append((model, api_key))
            raise RateLimitException(f"429 Quota Exhausted on {api_key}", retry_after=60.0)

        async def backup_chat(model, messages, api_key, **kwargs):
            call_log.append((model, api_key))
            return UnifiedChatResult(
                content="Success from backup model!",
                prompt_tokens=10,
                completion_tokens=5,
            )

        mock_primary.chat_completion.side_effect = primary_chat
        mock_backup.chat_completion.side_effect = backup_chat

        router_engine.providers["prov_primary"] = mock_primary
        router_engine.providers["prov_backup"] = mock_backup

        with patch.object(
            router_engine,
            "resolve_fallback_chain",
            return_value=[("prov_primary", "model-primary"), ("prov_backup", "model-backup")],
        ):
            res, p_resolved, m_resolved, meta = await router_engine.execute_chat(
                requested_model="test-stack",
                messages=[{"role": "user", "content": "hello"}],
            )

            # Assert primary was fully exhausted before cascading to backup
            self.assertEqual(p_resolved, "prov_backup")
            self.assertEqual(m_resolved, "model-backup")
            self.assertEqual(res.content, "Success from backup model!")

            # Call sequence MUST show: model-primary(k1) -> model-primary(k2) -> model-backup(k1)
            self.assertEqual(len(call_log), 3)
            self.assertEqual(call_log[0], ("model-primary", "key_prim_1"))
            self.assertEqual(call_log[1], ("model-primary", "key_prim_2"))
            self.assertEqual(call_log[2], ("model-backup", "key_back_1"))

            attempts = meta["candidates_attempted"]
            self.assertEqual(len(attempts), 3)
            self.assertEqual(attempts[0]["status"], "rate_limited")
            self.assertEqual(attempts[1]["status"], "rate_limited")
            self.assertEqual(attempts[2]["status"], "success")

    async def test_stream_exhausts_keys_before_model_failover(self):
        """Verify stream initiation also exhausts keys on current model before failover."""
        from unittest.mock import MagicMock
        from app.core.key_pool import key_manager

        key_manager.register_provider("prov_primary", ["key_stream_1", "key_stream_2"])
        key_manager.register_provider("prov_backup", ["key_back_1"])

        mock_primary = MagicMock()
        call_log = []

        def primary_stream(model, messages, api_key, **kwargs):
            call_log.append((model, api_key))
            if api_key == "key_stream_1":
                raise RateLimitException("429 Stream rate limited", retry_after=30.0)

            async def gen():
                yield {"text": "Stream chunk 1", "finish_reason": None}
                yield {"text": "Stream chunk 2", "finish_reason": "stop"}

            return gen()

        mock_primary.stream_chat_completion.side_effect = primary_stream
        router_engine.providers["prov_primary"] = mock_primary

        with patch.object(
            router_engine,
            "resolve_fallback_chain",
            return_value=[("prov_primary", "model-primary")],
        ):
            stream_gen, p_resolved, m_resolved, meta = await router_engine.execute_stream(
                requested_model="test-stack",
                messages=[{"role": "user", "content": "stream test"}],
            )

            self.assertEqual(p_resolved, "prov_primary")
            self.assertEqual(m_resolved, "model-primary")

            chunks = []
            async for c in stream_gen:
                chunks.append(c)

            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[0]["text"], "Stream chunk 1")
            self.assertEqual(len(call_log), 2)
            self.assertEqual(call_log[0], ("model-primary", "key_stream_1"))
            self.assertEqual(call_log[1], ("model-primary", "key_stream_2"))


if __name__ == "__main__":
    unittest.main()
