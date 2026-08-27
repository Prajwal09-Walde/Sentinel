import os
import math
from typing import Dict, Any, Optional
from contextvars import ContextVar
from unittest.mock import patch
from pydantic import BaseModel

class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float

# Thread/Async-safe ContextVars
active_tracker_var: ContextVar[Optional["CostTracker"]] = ContextVar("active_tracker", default=None)
active_call_type_var: ContextVar[str] = ContextVar("active_call_type", default="generation")

# Per-model pricing details (Price per Million Tokens)
# Source: OpenAI official pricing, easily updatable.
PRICING_CONFIG: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {
        "prompt_tokens_price_per_million": 0.150,
        "completion_tokens_price_per_million": 0.600,
    },
    "gpt-4o": {
        "prompt_tokens_price_per_million": 5.000,
        "completion_tokens_price_per_million": 15.000,
    },
    "gpt-3.5-turbo": {
        "prompt_tokens_price_per_million": 0.500,
        "completion_tokens_price_per_million": 1.500,
    },
    "gpt-4": {
        "prompt_tokens_price_per_million": 30.000,
        "completion_tokens_price_per_million": 60.000,
    },
    "default": {
        "prompt_tokens_price_per_million": 0.150,
        "completion_tokens_price_per_million": 0.600,
    }
}


def calculate_token_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate the cost in USD based on model pricing.
    
    Args:
        model_name: Name of the LLM model used.
        prompt_tokens: Number of tokens in the prompt/input.
        completion_tokens: Number of tokens in the completion/output.
        
    Returns:
        Calculated cost in USD.
    """
    model = model_name.lower()
    matched_config = PRICING_CONFIG["default"]
    
    for key, config in PRICING_CONFIG.items():
        if key in model:
            matched_config = config
            break
            
    prompt_cost = (prompt_tokens / 1_000_000.0) * matched_config["prompt_tokens_price_per_million"]
    completion_cost = (completion_tokens / 1_000_000.0) * matched_config["completion_tokens_price_per_million"]
    return prompt_cost + completion_cost


class CostTracker:
    """Context manager to wrap request lifecycle and automatically track LLM call tokens/costs."""
    
    def __init__(self) -> None:
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.cost_by_call_type: Dict[str, float] = {
            "generation": 0.0,
            "ragas_eval": 0.0,
            "deepeval_eval": 0.0,
            "guardrail_check": 0.0
        }
        self.tokens_by_call_type: Dict[str, int] = {
            "generation": 0,
            "ragas_eval": 0,
            "deepeval_eval": 0,
            "guardrail_check": 0
        }
        self._patches = []
        self._token = None

    def __enter__(self) -> "CostTracker":
        self._token = active_tracker_var.set(self)
        self._setup_patches()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._token:
            active_tracker_var.reset(self._token)
        for p in self._patches:
            p.stop()

    def _setup_patches(self) -> None:
        """Apply monkey-patches to openai chat completions to record token usage."""
        # 1. Patch sync Completions.create
        try:
            from openai.resources.chat.completions import Completions
            original_sync_create = Completions.create
            
            def hooked_sync_create(instance, *args, **kwargs):
                response = original_sync_create(instance, *args, **kwargs)
                try:
                    model = kwargs.get("model", "unknown")
                    if hasattr(response, "usage") and response.usage:
                        prompt_tokens = response.usage.prompt_tokens
                        completion_tokens = response.usage.completion_tokens
                        self.log_transaction(model, prompt_tokens, completion_tokens)
                except Exception:
                    pass
                return response

            p_sync = patch.object(Completions, "create", hooked_sync_create)
            p_sync.start()
            self._patches.append(p_sync)
        except Exception:
            pass

        # 2. Patch async Completions.create
        try:
            from openai.resources.chat.completions import AsyncCompletions
            original_async_create = AsyncCompletions.create
            
            async def hooked_async_create(instance, *args, **kwargs):
                response = await original_async_create(instance, *args, **kwargs)
                try:
                    model = kwargs.get("model", "unknown")
                    if hasattr(response, "usage") and response.usage:
                        prompt_tokens = response.usage.prompt_tokens
                        completion_tokens = response.usage.completion_tokens
                        self.log_transaction(model, prompt_tokens, completion_tokens)
                except Exception:
                    pass
                return response

            p_async = patch.object(AsyncCompletions, "create", hooked_async_create)
            p_async.start()
            self._patches.append(p_async)
        except Exception:
            pass

    def log_transaction(self, model_name: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Log input/output tokens and update total accumulated cost.
        
        Args:
            model_name: Name of the LLM model used.
            prompt_tokens: Number of tokens in the prompt/input.
            completion_tokens: Number of tokens in the completion/output.
        """
        call_type = active_call_type_var.get()
        cost = calculate_token_cost(model_name, prompt_tokens, completion_tokens)
        tokens = prompt_tokens + completion_tokens
        
        self.total_tokens += tokens
        self.total_cost_usd += cost
        
        if call_type in self.cost_by_call_type:
            self.cost_by_call_type[call_type] += cost
            self.tokens_by_call_type[call_type] += tokens

    def get_summary(self) -> Dict[str, Any]:
        """Return total tokens used and total cost.
        
        Returns:
            Dict containing accumulated total_tokens, total_cost_usd, and transaction breakdown.
        """
        return {
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "cost_by_call_type": self.cost_by_call_type.copy(),
            "tokens_by_call_type": self.tokens_by_call_type.copy()
        }
