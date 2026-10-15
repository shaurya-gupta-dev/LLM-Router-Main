"""Intelligent Hybrid Query Classifier for LLM Router."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Heuristic regex patterns
CODE_PATTERNS = [
    re.compile(r"```"),                                                  # Code blocks
    re.compile(r"\b(def|class|import|async def|return)\b"),               # Python
    re.compile(r"\b(function|const|let|var|interface|typeof)\b"),         # JS/TS
    re.compile(r"\b(public|private|protected|void|static)\b"),           # Java/C#/C++
    re.compile(r"\b(SELECT\s+.+\s+FROM|INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM)\b", re.I),  # SQL
    re.compile(r"\b(Traceback|NullPointerException|TypeError|SyntaxError|ReferenceError)\b"),       # Errors
    re.compile(r"\b(git|npm|pip|docker|docker-compose|kubectl|curl|postgres|mysql|redis|mongodb|aws|k8s)\b", re.I),
    re.compile(r"\b(refactor|debug|compile|regex|algorithm|unit test|database)\b", re.I),
]

CASUAL_PATTERNS = [
    re.compile(r"\b(hi|hello|hey|greetings|hola|sup|good morning|good evening|good afternoon)\b", re.I),
    re.compile(r"\b(who are you|what is your name|how are you|tell me a joke|thank you|thanks)\b", re.I),
    re.compile(r"\b(summarize|summarise|translate to|write an email|paraphrase|proofread)\b", re.I),
]

REASONING_PATTERNS = [
    re.compile(r"\b(system design|architecture|tradeoffs|trade-offs|distributed systems)\b", re.I),
    re.compile(r"\b(mathematical proof|derive|step by step reasoning|formal verification)\b", re.I),
    re.compile(r"\b(concurrency|deadlock|race condition|consensus|paxos|raft)\b", re.I),
]


class QueryClassifier:
    """Classifies user queries into CASUAL, CODING, or REASONING tiers."""

    def __init__(self):
        self.micro_client = None

    def _extract_text(self, messages: List[Dict[str, Any]]) -> str:
        """Extract user prompt text from chat messages."""
        user_texts = []
        for m in reversed(messages):
            if m.get("role") == "user":
                c = m.get("content", "")
                if isinstance(c, str):
                    user_texts.append(c)
                elif isinstance(c, list):
                    for part in c:
                        if isinstance(part, dict) and part.get("type") == "text":
                            user_texts.append(part.get("text", ""))
                break
        return " ".join(user_texts).strip()

    def classify_heuristic(self, text: str) -> Optional[str]:
        """Sub-millisecond fast-path heuristic classifier."""
        if not text:
            return "CASUAL"

        # Check reasoning first (if explicit architecture/math)
        for pat in REASONING_PATTERNS:
            if pat.search(text):
                return "REASONING"

        # Check code patterns
        for pat in CODE_PATTERNS:
            if pat.search(text):
                return "CODING"

        # Check casual / general chat patterns
        for pat in CASUAL_PATTERNS:
            if pat.search(text):
                return "CASUAL"

        return None

    async def classify(
        self,
        messages: List[Dict[str, Any]],
        micro_infer_func: Optional[Any] = None,
    ) -> Tuple[str, str, float]:
        """
        Classifies incoming prompt.
        Returns: (tier, method, latency_ms)
          tier: 'CASUAL' | 'CODING' | 'REASONING'
          method: 'heuristic' | 'micro_llm' | 'fallback'
        """
        start_t = time.perf_counter()
        text = self._extract_text(messages)

        # 1. Fast-path heuristic (<0.5ms)
        h_tier = self.classify_heuristic(text)
        if h_tier:
            elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)
            return h_tier, "heuristic", elapsed_ms

        # 2. Stage 2: Ultra-Fast Micro-Classifier (Cerebras/Groq, ~80-120ms)
        if micro_infer_func:
            try:
                micro_prompt = (
                    "Classify this user request into exactly one letter:\n"
                    "G = General casual conversation, small talk, or simple Q&A\n"
                    "C = Code writing, debugging, refactoring, or programming syntax\n"
                    "R = Complex reasoning, system architecture, hard logic, or mathematics\n\n"
                    f"User Request: {text[:300]}\n\n"
                    "Answer with only the single uppercase letter (G, C, or R):"
                )
                pred = await micro_infer_func(micro_prompt)
                pred_letter = (pred or "").strip().upper()
                elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)

                if "C" in pred_letter:
                    return "CODING", "micro_llm", elapsed_ms
                elif "R" in pred_letter:
                    return "REASONING", "micro_llm", elapsed_ms
                elif "G" in pred_letter:
                    return "CASUAL", "micro_llm", elapsed_ms
            except Exception as e:
                logger.warning("Micro-classifier failed: %s. Using default.", e)

        # 3. Default fallback based on prompt length
        elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)
        words = text.split()
        if len(words) > 80:
            return "REASONING", "length_fallback", elapsed_ms
        return "CODING", "default_fallback", elapsed_ms


classifier = QueryClassifier()
