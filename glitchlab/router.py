"""
GLITCHLAB Router — Vendor-Agnostic Model Abstraction

Routes agent calls through LiteLLM so agents never know
which vendor is backing them. Handles budget tracking,
retries, and structured logging.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import litellm
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from glitchlab.config_loader import GlitchLabConfig


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

    @property
    def tokens_remaining(self) -> int:
        """Calculate the number of tokens remaining in the budget.

        Returns:
            int: The number of tokens remaining.
        """
        return max(0, self.max_tokens - self.usage.total_tokens)

    @property
    def dollars_remaining(self) -> float:
        """Calculate the number of dollars remaining in the budget.

        Returns:
            float: The number of dollars remaining.
        """
        return max(0.0, self.max_dollars - self.usage.estimated_cost)

    @property
    def budget_exceeded(self) -> bool:
        """Check if the budget has been exceeded.

        Returns:
            bool: True if the budget is exceeded, False otherwise.
        """
        return self.usage.total_tokens >= self.max_tokens or self.usage.estimated_cost >= self.max_dollars

    def record(self, response: Any) -> None:
        """Record usage from a LiteLLM response.

        Extracts token usage from the response object and updates the internal
        counters for prompt, completion, and total tokens. It also calculates
        the estimated cost using LiteLLM's cost calculator and adds it to the
        total estimated cost.

        Args:
            response (Any): The response object returned by LiteLLM.
        """
        usage = getattr(response, "usage", None)
        if usage:
            self.usage.prompt_tokens += getattr(usage, "prompt_tokens", 0)
            self.usage.completion_tokens += getattr(usage, "completion_tokens", 0)
            self.usage.total_tokens += getattr(usage, "total_tokens", 0)

        try:
            cost = litellm.completion_cost(completion_response=response)
            self.usage.estimated_cost += cost
        except Exception:
            pass

        self.usage.call_count += 1

    def summary(self) -> dict:
        """Generate a summary of current budget usage.

        Returns:
            dict: A dictionary containing total tokens used, estimated cost,
                call count, tokens remaining, and dollars remaining.
        """
        return {
            "total_tokens": self.usage.total_tokens,
            "estimated_cost": round(self.usage.estimated_cost, 4),
            "call_count": self.usage.call_count,
            "tokens_remaining": self.tokens_remaining,
            "dollars_remaining": round(self.dollars_remaining, 4),
        }


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
) -> dict[str, Any]:
    """
    Build LiteLLM kwargs with per-model param filtering.
    Different model families support different parameters.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }

    # GPT-5 and o-series models don't support arbitrary temperature
    if not _is_gpt5_model(model) and not _is_o_series_model(model):
        kwargs["temperature"] = temperature

    if response_format:
        kwargs["response_format"] = response_format

    return kwargs


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class AgentMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class RouterResponse(BaseModel):
    content: str
    model: str
    tokens_used: int = 0
    cost: float = 0.0
    latency_ms: int = 0


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
        self._role_model_map = {
            "planner": config.routing.planner,
            "implementer": config.routing.implementer,
            "debugger": config.routing.debugger,
            "security": config.routing.security,
            "release": config.routing.release,
            "archivist": config.routing.archivist,
        }

        litellm.suppress_debug_info = True

    def resolve_model(self, role: str) -> str:
        """Resolve agent role to a specific model string.

        Args:
            role (str): The agent role name (e.g., 'planner', 'implementer').

        Returns:
            str: The model string associated with the given role.

        Raises:
            ValueError: If the provided role is not found in the role-to-model mapping.
        """
        model = self._role_model_map.get(role)
        if not model:
            raise ValueError(f"Unknown agent role: {role}. Known: {list(self._role_model_map)}")
        return model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def complete(
        self,
        role: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> RouterResponse:
        """Send a completion request through LiteLLM.

        Routes the request to the appropriate model based on the agent role.
        It enforces budget constraints before making the call. After the call,
        it records the token usage and calculates the cost using the BudgetTracker.

        Args:
            role (str): Agent role name (planner, implementer, etc.).
            messages (list[dict[str, str]]): Standard chat messages [{"role": ..., "content": ...}].
            temperature (float, optional): Sampling temperature. Defaults to 0.2.
                Dropped automatically for models that don't support it.
            max_tokens (int, optional): Max response tokens. Defaults to 4096.
            response_format (dict | None, optional): Optional JSON schema for structured output. Defaults to None.

        Returns:
            RouterResponse: A structured response containing the content, model used,
                tokens used, estimated cost, and latency.

        Raises:
            BudgetExceededError: If the budget limits for tokens or dollars are exceeded.
        """
        if self.budget.budget_exceeded:
            raise BudgetExceededError(
                f"Budget exceeded: {self.budget.summary()}"
            )

        model = self.resolve_model(role)
        start = time.monotonic()

        logger.debug(f"[ROUTER] {role} → {model} ({len(messages)} messages)")

        kwargs = _build_kwargs(model, messages, temperature, max_tokens, response_format)

        response = litellm.completion(**kwargs)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        self.budget.record(response)

        content = response.choices[0].message.content or ""

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
        )


class BudgetExceededError(Exception):
    pass
