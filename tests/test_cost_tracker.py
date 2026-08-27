import pytest
from unittest.mock import patch, MagicMock
from openai import OpenAI
from eval_harness.cost_tracker import (
    CostTracker,
    calculate_token_cost,
    active_call_type_var
)

def test_calculate_token_cost():
    # gpt-4o-mini cost calculation:
    # prompt: 0.150 / M -> 1000 tokens = 0.00015 USD
    # completion: 0.600 / M -> 500 tokens = 0.0003 USD
    # total: 0.00045 USD
    cost = calculate_token_cost("gpt-4o-mini", 1000, 500)
    assert pytest.approx(cost, abs=1e-6) == 0.00045

    # gpt-4 cost calculation:
    # prompt: 30.00 / M -> 1000 tokens = 0.030 USD
    # completion: 60.00 / M -> 500 tokens = 0.030 USD
    # total: 0.060 USD
    cost_gpt4 = calculate_token_cost("gpt-4", 1000, 500)
    assert pytest.approx(cost_gpt4, abs=1e-6) == 0.060


@patch("openai.resources.chat.completions.Completions.create")
def test_cost_tracker_lifecycle(mock_openai_create):
    # Prepare mock responses with custom usage details
    # We will configure mock_openai_create to return different tokens depending on model input
    def side_effect(*args, **kwargs):
        model = kwargs.get("model", "unknown")
        response = MagicMock()
        
        if model == "gpt-4":
            response.usage.prompt_tokens = 500
            response.usage.completion_tokens = 200
        else:
            # default to gpt-4o-mini size
            response.usage.prompt_tokens = 1000
            response.usage.completion_tokens = 200
            
        return response

    mock_openai_create.side_effect = side_effect

    with CostTracker() as ct:
        client = OpenAI(api_key="mock-key")
        
        # 1. Generation call (uses gpt-4)
        client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": "hello"}])
        
        # 2. Guardrail check (uses gpt-4o-mini)
        token = active_call_type_var.set("guardrail_check")
        client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "guardrail"}])
        active_call_type_var.reset(token)

        # 3. Ragas evaluation (uses gpt-4o-mini)
        token = active_call_type_var.set("ragas_eval")
        client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "ragas"}])
        active_call_type_var.reset(token)

    summary = ct.get_summary()

    # Verify token totals
    # Generation: 500 + 200 = 700
    # Guardrails: 1000 + 200 = 1200
    # Ragas: 1000 + 200 = 1200
    # Total = 3100
    assert summary["total_tokens"] == 3100
    assert ct.tokens_by_call_type["generation"] == 700
    assert ct.tokens_by_call_type["guardrail_check"] == 1200
    assert ct.tokens_by_call_type["ragas_eval"] == 1200

    # Verify cost totals
    # Generation (gpt-4): 500 * 30.00 / 1M + 200 * 60.00 / 1M = 0.015 + 0.012 = 0.027 USD
    # Guardrail (gpt-4o-mini): 1000 * 0.15 / 1M + 200 * 0.60 / 1M = 0.00015 + 0.00012 = 0.00027 USD
    # Ragas (gpt-4o-mini): 1000 * 0.15 / 1M + 200 * 0.60 / 1M = 0.00027 USD
    # Total cost = 0.027 + 0.00027 + 0.00027 = 0.02754 USD
    assert pytest.approx(summary["total_cost_usd"], abs=1e-6) == 0.02754
    assert pytest.approx(ct.cost_by_call_type["generation"], abs=1e-6) == 0.027


@patch("openai.resources.chat.completions.Completions.create")
def test_eval_overhead_ratio(mock_openai_create):
    """Regression test ensuring evaluation harness overhead doesn't exceed 3x raw generation cost."""
    
    # Mocking call responses:
    # Raw generation is run on gpt-4o (expensive model)
    # Evaluation/guardrails are run on gpt-4o-mini (cheap model)
    def side_effect(*args, **kwargs):
        model = kwargs.get("model", "unknown")
        response = MagicMock()
        
        if model == "gpt-4o":
            response.usage.prompt_tokens = 600
            response.usage.completion_tokens = 300
        elif model == "gpt-4o-mini":
            # Evaluators use fewer tokens or cheaper rates
            response.usage.prompt_tokens = 1000
            response.usage.completion_tokens = 200
        else:
            response.usage.prompt_tokens = 100
            response.usage.completion_tokens = 10
            
        return response

    mock_openai_create.side_effect = side_effect

    with CostTracker() as ct:
        client = OpenAI(api_key="mock-key")
        
        # 1. Generation call (gpt-4o)
        client.chat.completions.create(model="gpt-4o", messages=[])
        
        # 2. Guardrail check (gpt-4o-mini)
        token = active_call_type_var.set("guardrail_check")
        client.chat.completions.create(model="gpt-4o-mini", messages=[])
        active_call_type_var.reset(token)

        # 3. Ragas evaluation (gpt-4o-mini)
        token = active_call_type_var.set("ragas_eval")
        client.chat.completions.create(model="gpt-4o-mini", messages=[])
        active_call_type_var.reset(token)

        # 4. DeepEval evaluation (gpt-4o-mini)
        token = active_call_type_var.set("deepeval_eval")
        client.chat.completions.create(model="gpt-4o-mini", messages=[])
        active_call_type_var.reset(token)

    # Calculate overhead
    gen_cost = ct.cost_by_call_type["generation"]
    overhead_cost = (
        ct.cost_by_call_type["guardrail_check"] +
        ct.cost_by_call_type["ragas_eval"] +
        ct.cost_by_call_type["deepeval_eval"]
    )
    
    # Assert overhead <= 3x generation cost
    assert gen_cost > 0.0
    assert overhead_cost <= 3.0 * gen_cost
    
    # Print the ratio for visibility (e.g. 0.00081 / 0.0075 = ~10.8% overhead)
    print(f"\nEvaluation overhead ratio: {overhead_cost / gen_cost:.4f}x generation cost.")
