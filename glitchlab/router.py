"""
GLITCHLAB Router — Vendor-Agnostic Model Abstraction (v3.0 — Semantic Snipping)

Routes agent calls through LiteLLM so agents never know
which vendor is backing them. Handles budget tracking,
retries, structured logging, and automatic 503 failover.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import litellm
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type

from glitchlab.config_loader import GlitchLabConfig


class BudgetExceededError(Exception):
    pass


# ---------------------------------------------------------------------------
# Usage Tracking
# ---------------------------------------------------------------------------

@dataclass
class UsageRecord:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    call_count: int = 0


@dataclass
class BudgetTracker:
    """Tracks token + dollar spend per task."""
    max_tokens: int = 150_000
    max_dollars: float = 10.0
    usage: UsageRecord = field(default_factory=UsageRecord)
    role_usage: dict[str, int] = field(default_factory=dict)

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.max_tokens - self.usage.total_tokens)

    @property
    def dollars_remaining(self) -> float:
        return max(0.0, self.max_dollars - self.usage.estimated_cost)

    @property
    def budget_exceeded(self) -> bool:
        return self.usage.total_tokens >= self.max_tokens or self.usage.estimated_cost >= self.max_dollars

    def record(self, response: Any, role: str) -> None:
        """Record usage from a LiteLLM response."""
        usage = getattr(response, "usage", None)
        total_tokens = getattr(usage, "total_tokens", 0) if usage else 0
        if usage:
            self.usage.prompt_tokens += getattr(usage, "prompt_tokens", 0)
            self.usage.completion_tokens += getattr(usage, "completion_tokens", 0)
            self.usage.total_tokens += total_tokens

        try:
            cost = litellm.completion_cost(completion_response=response)
            self.usage.estimated_cost += cost
        except Exception:
            pass

        self.usage.call_count += 1
        self.role_usage[role] = self.role_usage.get(role, 0) + total_tokens

    def summary(self) -> dict:
        return {
            "total_tokens": self.usage.total_tokens,
            "estimated_cost": round(self.usage.estimated_cost, 4),
            "call_count": self.usage.call_count,
            "tokens_remaining": self.tokens_remaining,
            "dollars_remaining": round(self.dollars_remaining, 4),
        }


# ---------------------------------------------------------------------------
# Context Monitor (v3 — Semantic Snipping)
# ---------------------------------------------------------------------------

# Regex to extract file paths from message content (covers tool args, diffs, references)
_FILE_REF_RE = re.compile(
    r'(?:'
    r'(?:^|[\s"\':,{])('                   # leading context
    r'[a-zA-Z0-9_./\-]+'                   # path body
    r'\.(?:py|rs|tsx|ts|jsx|js|go|java'     # code extensions
    r'|toml|json|yaml|yml|md|cfg|ini)'      # config extensions
    r')'
    r'(?=$|[\s"\':,}\]\)])'                # must end at boundary (not mid-word)
    r')',
    re.MULTILINE,
)


def _extract_file_refs(content: str) -> set[str]:
    """Extract file path references from message text."""
    if not isinstance(content, str):
        return set()
    refs: set[str] = set()
    for match in _FILE_REF_RE.findall(content):
        # Normalize: strip leading ./ or /
        clean = match.lstrip("./")
        if clean and "/" in clean or "." in clean:
            refs.add(clean)
    return refs


def _estimate_tokens(content: str) -> int:
    """Fast character-based token estimate (avoids tokenizer overhead per-message)."""
    return len(content) // 4 if content else 0


class ContextMonitor:
    """
    Protects the LLM's output headroom by proactively snipping
    input context before the call if it gets too large.

    v3: Semantic snipping — scores messages by relevance to active files
    and drops messages about already-committed files first, preserving
    context the agent actually needs under token pressure.
    """

    def __init__(self, safe_headroom_tokens: int = 8192):
        self.safe_headroom = safe_headroom_tokens
        # Semantic state: updated by controller between phases
        self._active_files: set[str] = set()
        self._committed_files: set[str] = set()
        self._symbol_names: set[str] = set()

    def update_file_state(
        self,
        active_files: list[str] | None = None,
        committed_files: list[str] | None = None,
        symbol_names: list[str] | None = None,
    ) -> None:
        """
        Update the semantic state used for relevance scoring.

        Args:
            active_files: Files currently being modified / in scope.
            committed_files: Files already committed (safe to deprioritize).
            symbol_names: Symbol names from the index that are in scope.
        """
        if active_files is not None:
            self._active_files = {f.lstrip("./") for f in active_files}
        if committed_files is not None:
            self._committed_files = {f.lstrip("./") for f in committed_files}
        if symbol_names is not None:
            self._symbol_names = set(symbol_names)

    def _score_message(self, msg: dict, position: int, total: int) -> float:
        """
        Score a message's relevance (higher = more important to keep).

        Scoring factors:
          - References to active files:    +10 per file
          - References to committed files: -5 per file (safe to drop)
          - References to in-scope symbols: +3 per symbol
          - Recency bonus:                 0-5 (linear by position)
          - Role bonuses:                  tool results +2, errors +8
        """
        content = str(msg.get("content", ""))
        role = msg.get("role", "")
        score = 0.0

        # Recency: newer messages score higher (0 to 5)
        score += 5.0 * (position / max(total, 1))

        # File reference scoring
        refs = _extract_file_refs(content)
        if refs and (self._active_files or self._committed_files):
            for ref in refs:
                ref_norm = ref.lstrip("./")
                if any(ref_norm.endswith(a) or a.endswith(ref_norm) for a in self._active_files):
                    score += 10.0
                elif any(ref_norm.endswith(c) or c.endswith(ref_norm) for c in self._committed_files):
                    score -= 5.0

        # Symbol reference scoring
        if self._symbol_names:
            content_lower = content.lower()
            for sym in self._symbol_names:
                if sym.lower() in content_lower:
                    score += 3.0

        # Role-based bonuses
        if role == "tool":
            score += 2.0
            # Error messages are high-value context
            if any(kw in content.lower() for kw in ("error", "traceback", "failed", "exception")):
                score += 8.0

        return score

    def enforce_headroom(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
    ) -> list[dict[str, str]]:
        # 1. Determine model context window
        try:
            model_info = litellm.get_model_info(model)
            max_window = model_info.get("max_input_tokens") or model_info.get("max_tokens") or 128000
        except Exception:
            max_window = 128000

        # 2. Calculate hard limit for input
        target_output = max_tokens or self.safe_headroom
        input_limit = max_window - target_output - (self.safe_headroom // 2)

        # 3. Count current tokens
        try:
            current_tokens = litellm.token_counter(model=model, messages=messages)
        except Exception:
            current_tokens = sum(_estimate_tokens(str(m.get("content", ""))) for m in messages)

        if current_tokens <= input_limit:
            return messages

        has_semantic_state = bool(self._active_files or self._committed_files or self._symbol_names)

        if has_semantic_state:
            return self._semantic_snip(messages, current_tokens, input_limit)
        else:
            # Fallback: chronological snipping (v2 behavior)
            return self._chronological_snip(messages, current_tokens, input_limit)

    def _semantic_snip(
        self,
        messages: list[dict[str, str]],
        current_tokens: int,
        input_limit: int,
    ) -> list[dict[str, str]]:
        """
        Drop or compress messages by semantic relevance score.
        Low-scoring messages (about committed files, old exploration)
        are removed first. High-scoring messages (active files, errors)
        are preserved.
        """
        tokens_to_shed = current_tokens - input_limit

        logger.warning(
            f"⚠️ [CONTEXT] Semantic snip: need to shed ~{tokens_to_shed} tokens. "
            f"Active files: {len(self._active_files)}, "
            f"Committed files: {len(self._committed_files)}"
        )

        # Separate protected (system, last 2 messages) from candidates
        system_msgs: list[tuple[int, dict]] = []
        candidate_msgs: list[tuple[int, dict, float]] = []

        total = len(messages)
        # Always protect the last 2 non-system messages (most recent exchange)
        protected_tail = max(2, 0)

        non_system_count = sum(1 for m in messages if m.get("role") != "system")
        non_system_seen = 0

        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                system_msgs.append((i, msg))
            else:
                non_system_seen += 1
                tail_position = non_system_count - non_system_seen
                if tail_position < protected_tail:
                    # Protect last 2 messages
                    system_msgs.append((i, msg))
                else:
                    score = self._score_message(msg, i, total)
                    candidate_msgs.append((i, msg, score))

        # Sort candidates by score ascending (lowest relevance first = drop first)
        candidate_msgs.sort(key=lambda x: x[2])

        tokens_shed = 0
        kept_candidates: list[tuple[int, dict]] = []

        for idx, msg, score in candidate_msgs:
            msg_tokens = _estimate_tokens(str(msg.get("content", "")))

            if tokens_shed < tokens_to_shed:
                # Drop or aggressively compress this message
                if score <= 0:
                    # Negative/zero score: drop entirely (replace with marker)
                    marker = {
                        "role": msg.get("role", "user"),
                        "content": f"[SNIPPED — resolved context about committed files]",
                    }
                    # Preserve tool_call_id if present (required for tool messages)
                    if msg.get("tool_call_id"):
                        marker["tool_call_id"] = msg["tool_call_id"]
                    if msg.get("name"):
                        marker["name"] = msg["name"]
                    tokens_shed += msg_tokens - _estimate_tokens(marker["content"])
                    kept_candidates.append((idx, marker))
                else:
                    # Positive but low score: truncate to tail
                    content = str(msg.get("content", ""))
                    if len(content) > 300:
                        keep_ratio = max(0.15, 1.0 - (tokens_to_shed - tokens_shed) / max(msg_tokens, 1))
                        target_len = int(len(content) * keep_ratio)
                        target_len = max(target_len, 150)  # Always keep at least 150 chars
                        new_content = "\n...[SNIPPED BY SEMANTIC CONTEXT MONITOR]...\n" + content[-target_len:]
                        tokens_shed += _estimate_tokens(content) - _estimate_tokens(new_content)
                        new_msg = {**msg, "content": new_content}
                        kept_candidates.append((idx, new_msg))
                    else:
                        kept_candidates.append((idx, msg))
            else:
                # Budget met — keep remaining messages intact
                kept_candidates.append((idx, msg))

        # Reassemble in original order
        all_msgs = system_msgs + kept_candidates
        all_msgs.sort(key=lambda x: x[0])

        result = [msg for _, msg in all_msgs]

        logger.info(
            f"[CONTEXT] Semantic snip complete: shed ~{tokens_shed} tokens, "
            f"kept {len(result)}/{len(messages)} messages"
        )

        return result

    def _chronological_snip(
        self,
        messages: list[dict[str, str]],
        current_tokens: int,
        input_limit: int,
    ) -> list[dict[str, str]]:
        """Fallback: uniform ratio-based truncation (original v2 behavior)."""
        logger.warning(
            f"⚠️ [CONTEXT] Token pressure high ({current_tokens} > {input_limit}). "
            "Snipping oldest context to prevent JSON truncation..."
        )

        snip_ratio = max(0.15, input_limit / current_tokens)

        new_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                new_messages.append(msg)
            else:
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > 500:
                    target_len = int(len(content) * snip_ratio)
                    content = "\n...[TRUNCATED BY CONTEXT MONITOR]...\n" + content[-target_len:]
                new_messages.append({"role": msg.get("role"), "content": content})

        return new_messages


# ---------------------------------------------------------------------------
# Model capability helpers
# ---------------------------------------------------------------------------

def _is_gpt5_model(model: str) -> bool:
    """GPT-5 family models have restricted parameter support."""
    normalized = model.lower().replace("openai/", "")
    return normalized.startswith("gpt-5")


def _is_o_series_model(model: str) -> bool:
    """OpenAI o-series reasoning models don't support temperature."""
    normalized = model.lower().replace("openai/", "")
    return normalized.startswith("o1") or normalized.startswith("o3") or normalized.startswith("o4")


def _build_kwargs(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    response_format: dict | None,
    tools: list[dict] | None = None,
    **kwargs
) -> dict[str, Any]:
    kwargs_dict: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "timeout": 120,
    }

    if not _is_gpt5_model(model) and not _is_o_series_model(model):
        kwargs_dict["temperature"] = temperature

    if response_format:
        kwargs_dict["response_format"] = response_format
        
    if tools:
        kwargs_dict["tools"] = tools

    # Pass through extra parameters like tool_choice
    for k, v in kwargs.items():
        kwargs_dict[k] = v

    return kwargs_dict


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class AgentMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class RouterResponse(BaseModel):
    content: str | None = None  # LLMs return None for content when calling tools
    model: str
    tokens_used: int = 0
    cost: float = 0.0
    latency_ms: int = 0
    tool_calls: Any | None = None  # Attach the raw LiteLLM tool calls


class Router:
    """
    Vendor-agnostic model router.

    Agents call `router.complete(role, messages)`.
    The router resolves the model, enforces budget, and returns structured output.
    """

    def __init__(self, config: GlitchLabConfig):
        self.config = config
        self.budget = BudgetTracker(
            max_tokens=config.limits.max_tokens_per_task,
            max_dollars=config.limits.max_dollars_per_task,
        )
        self.context_monitor = ContextMonitor(safe_headroom_tokens=8192)

    def update_semantic_state(
        self,
        active_files: list[str] | None = None,
        committed_files: list[str] | None = None,
        symbol_names: list[str] | None = None,
    ) -> None:
        """Forward semantic state to the context monitor for smarter snipping."""
        self.context_monitor.update_file_state(
            active_files=active_files,
            committed_files=committed_files,
            symbol_names=symbol_names,
        )
        
        self._role_model_map = {
            "planner": config.routing.planner,
            "implementer": config.routing.implementer,
            "debugger": config.routing.debugger,
            "security": config.routing.security,
            "release": config.routing.release,
            "archivist": config.routing.archivist,
            "testgen": config.routing.testgen,
            "scout": config.routing.scout,
            "critic": config.routing.critic,
        }

        litellm.suppress_debug_info = True

    def resolve_model(self, role: str) -> str:
        """Resolve agent role → model string."""
        model = self._role_model_map.get(role)
        if not model:
            raise ValueError(f"Unknown agent role: {role}. Known: {list(self._role_model_map)}")
        return model

    @retry(
        stop=stop_after_attempt(6), 
        wait=wait_exponential(min=2, max=60),
        retry=retry_if_not_exception_type(BudgetExceededError)
    )
    def complete(
        self,
        role: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        response_format: dict | None = None,
        tools: list[dict] | None = None,
        **kwargs
    ) -> RouterResponse:
        """
        Send a completion request through LiteLLM.

        Args:
            role: Agent role name (planner, implementer, etc.)
            messages: Standard chat messages [{"role": ..., "content": ...}]
            temperature: Sampling temperature (dropped automatically for models that don't support it)
            max_tokens: Max response tokens
            response_format: Optional JSON schema for structured output
            tools: Optional list of tools/functions the agent can call
        """
        if self.budget.budget_exceeded:
            raise BudgetExceededError(
                f"Budget exceeded: {self.budget.summary()}"
            )

        role_limits = {
            "planner": 0.15,
            "implementer": 0.60,
            "debugger": 0.30,
            "auditor": 0.10,
            "scout": 0.20,
            "security": 0.30,
            "release": 0.10,
            "archivist": 0.10,
            "critic": 0.05,
        }
        
        limit_ratio = role_limits.get(role, 0.50)
        role_token_limit = int(self.budget.max_tokens * limit_ratio)
        current_role_usage = self.budget.role_usage.get(role, 0)
        
        if current_role_usage >= role_token_limit:
            raise BudgetExceededError(
                f"Role budget exceeded for {role}: {current_role_usage} / {role_token_limit} tokens"
            )

        model = self.resolve_model(role)
        
        # V2: Enforce proactive context headroom
        safe_messages = self.context_monitor.enforce_headroom(messages, model, max_tokens)
        
        start = time.monotonic()

        logger.debug(f"[ROUTER] {role} → {model} ({len(safe_messages)} messages)")

        kwargs_dict = _build_kwargs(
            model, safe_messages, temperature, max_tokens, response_format, tools, **kwargs
        )

        try:
            response = litellm.completion(**kwargs_dict)
        except litellm.exceptions.ServiceUnavailableError:
            # Determine which fallback to use based on the primary model tier
            # Logic: If primary is a preview/pro model, use high_tier fallback.
            fallback_model = self.config.fallbacks.high_tier
            logger.warning(f"⚠️ [ROUTER] 503 Service Unavailable from {model}. Failing over to {fallback_model}...")
            
            # Rebuild kwargs for the fallback model
            kwargs_dict = _build_kwargs(
                fallback_model, safe_messages, temperature, max_tokens, response_format, tools, **kwargs
            )
            response = litellm.completion(**kwargs_dict)

        # Accuracy Fix: Place calculation here to capture total time spent including failover
        elapsed_ms = int((time.monotonic() - start) * 1000)

        self.budget.record(response, role)

        # Extract tool calls safely
        response_message = response.choices[0].message
        content = response_message.content
        tool_calls = getattr(response_message, "tool_calls", None)

        logger.debug(
            f"[ROUTER] {role} complete — "
            f"{self.budget.usage.total_tokens} tokens, "
            f"${self.budget.usage.estimated_cost:.4f}, "
            f"{elapsed_ms}ms"
        )

        return RouterResponse(
            content=content,
            model=model,
            tokens_used=getattr(response.usage, "total_tokens", 0),
            cost=self.budget.usage.estimated_cost,
            latency_ms=elapsed_ms,
            tool_calls=tool_calls
        )