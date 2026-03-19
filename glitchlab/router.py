"""
GLITCHLAB Router — Vendor-Agnostic Model Abstraction (v2.2)

Routes agent calls through LiteLLM so agents never know
which vendor is backing them. Handles budget tracking,
retries, structured logging, and automatic 503 failover.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import litellm
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type

from glitchlab.config_loader import GlitchLabConfig
from glitchlab.event_bus import bus


class BudgetExceededError(Exception):
    pass


class ContextOverflowError(Exception):
    """Raised when context shedding cannot recover from rate limits."""
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
            "role_usage": dict(self.role_usage),
        }


# ---------------------------------------------------------------------------
# Context Monitor (v2)
# ---------------------------------------------------------------------------

class ContextMonitor:
    """
    Protects the LLM's output headroom by proactively snipping 
    input context before the call if it gets too large.
    """
    def __init__(self, safe_headroom_tokens: int = 8192):
        # Always reserve this many tokens strictly for the model's response
        self.safe_headroom = safe_headroom_tokens

    def enforce_headroom(self, messages: list[dict[str, str]], model: str, max_tokens: int) -> list[dict[str, str]]:
        # 1. Determine model context window
        try:
            model_info = litellm.get_model_info(model)
            max_window = model_info.get("max_input_tokens") or model_info.get("max_tokens") or 128000
        except Exception:
            max_window = 128000 # Fallback to a safe default (e.g., standard GPT-4o window)
            
        # 2. Calculate our hard limit for the input prompt
        target_output = max_tokens or self.safe_headroom
        input_limit = max_window - target_output - (self.safe_headroom // 2)
        
        # 3. Count current tokens (fallback to fast character approximation if tokenizer fails)
        try:
            current_tokens = litellm.token_counter(model=model, messages=messages)
        except Exception:
            current_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4
            
        if current_tokens <= input_limit:
            return messages
            
        logger.warning(
            f"⚠️ [CONTEXT] Token pressure high ({current_tokens} > {input_limit}). "
            "Snipping oldest context to prevent JSON truncation..."
        )
        
        # 4. Snip logic: Reduce the length of non-system messages to fit
        snip_ratio = input_limit / current_tokens
        snip_ratio = max(0.15, snip_ratio) # Never truncate beyond 15% of original to retain some context
        
        new_messages = []
        _COMPRESSED_MARKERS = ("[Content compressed", "[Search results compressed")
        for msg in messages:
            if msg.get("role") == "system":
                # Never truncate system instructions
                new_messages.append(msg)
            else:
                content = msg.get("content", "")
                # Skip messages already compressed by agent-level compression
                if isinstance(content, str) and any(m in content for m in _COMPRESSED_MARKERS):
                    new_messages.append(msg)
                elif isinstance(content, str) and len(content) > 500:
                    target_len = int(len(content) * snip_ratio)
                    # Keep the end of the message (usually contains the most recent errors/instructions)
                    content = "\n...[TRUNCATED BY CONTEXT MONITOR]...\n" + content[-target_len:]
                    new_messages.append({"role": msg.get("role"), "content": content})
                else:
                    new_messages.append(msg)
                
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
        
        self._role_model_map = {
            "planner": config.routing.planner,
            "implementer": config.routing.implementer,
            "debugger": config.routing.debugger,
            "security": config.routing.security,
            "release": config.routing.release,
            "archivist": config.routing.archivist,
            "testgen": config.routing.testgen,
        }

        # Build fallback tier map from pipeline config
        self._role_fallback_tier = {
            step.agent_role: step.fallback_tier
            for step in config.pipeline
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
        retry=retry_if_not_exception_type((BudgetExceededError, ContextOverflowError))
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
            "security": 0.30,
            "release": 0.10,
            "archivist": 0.10,
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

        if len(safe_messages) != len(messages) or any(
            safe_messages[i].get("content") != messages[i].get("content")
            for i in range(min(len(safe_messages), len(messages)))
        ):
            bus.emit(
                event_type="context_monitor.snipped",
                payload={
                    "role": role,
                    "model": model,
                    "original_message_count": len(messages),
                    "snipped_message_count": len(safe_messages),
                },
                agent_id=role,
            )

        start = time.monotonic()

        logger.debug(f"[ROUTER] {role} → {model} ({len(safe_messages)} messages)")

        kwargs_dict = _build_kwargs(
            model, safe_messages, temperature, max_tokens, response_format, tools, **kwargs
        )

        bus.emit(
            event_type="llm.started",
            payload={"role": role, "model": model, "message_count": len(messages)},
            agent_id=role,
        )

        try:
            response = litellm.completion(**kwargs_dict)
        except litellm.exceptions.ServiceUnavailableError:
            # Select fallback tier from pipeline config (defaults to high)
            tier = self._role_fallback_tier.get(role, "high")
            fallback_model = (
                self.config.fallbacks.low_tier
                if tier == "low"
                else self.config.fallbacks.high_tier
            )
            model = fallback_model  # Update so events and RouterResponse reflect actual model
            logger.warning(
                f"⚠️ [ROUTER] 503 Service Unavailable. "
                f"Failing over to {fallback_model} (tier={tier})..."
            )

            # Rebuild kwargs for the fallback model
            kwargs_dict = _build_kwargs(
                fallback_model, safe_messages, temperature, max_tokens,
                response_format, tools, **kwargs
            )
            response = litellm.completion(**kwargs_dict)
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate" in error_str.lower()
            is_bad_request = "400" in error_str or "BadRequest" in error_str

            bus.emit(
                event_type="llm.error",
                payload={
                    "role": role,
                    "model": model,
                    "error": error_str,
                    "is_rate_limit": is_rate_limit,
                    "is_bad_request": is_bad_request,
                },
                agent_id=role,
            )

            if is_bad_request:
                # Bad request (likely broken tool pairing) — do not retry,
                # the same payload will fail every time.
                logger.error(
                    f"🛑 [ROUTER] BadRequest from API — not retrying. "
                    f"Likely broken tool_call/tool_result pairing. Error: {error_str[:200]}"
                )
                raise ContextOverflowError(
                    f"API rejected request (BadRequest). Context may be malformed: {error_str[:200]}"
                ) from e

            if is_rate_limit:
                logger.warning(
                    f"⚠️ [ROUTER] Rate limited. Will retry with backoff. Error: {error_str[:200]}"
                )

            raise

        # Accuracy Fix: Place calculation here to capture total time spent including failover
        elapsed_ms = int((time.monotonic() - start) * 1000)

        self.budget.record(response, role)

        bus.emit(
            event_type="llm.completed",
            payload={
                "role": role,
                "model": model,
                "tokens_prompt": response.usage.prompt_tokens if response.usage else 0,
                "tokens_completion": (
                    response.usage.completion_tokens if response.usage else 0
                ),
                "tokens_total": response.usage.total_tokens if response.usage else 0,
                "estimated_cost": round(self.budget.usage.estimated_cost, 4),
                "duration_ms": int(elapsed_ms),
            },
            agent_id=role,
        )

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