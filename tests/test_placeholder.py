import pytest
from eval_harness.metrics import EvalResult, evaluate_faithfulness
from eval_harness.guardrails import check_hallucination
from eval_harness.cost_tracker import TokenUsage, CostTracker
from eval_harness.tracing import TraceManager, trace_step
from eval_harness.runner import EvalRunner

def test_imports_and_models():
    # Verify we can instantiate the EvalResult model with all fields
    eval_res = EvalResult(
        query="What is RAG?",
        retrieved_contexts=["RAG stands for Retrieval-Augmented Generation."],
        generated_answer="RAG stands for Retrieval-Augmented Generation.",
        ground_truth="RAG is Retrieval-Augmented Generation.",
        faithfulness_score=1.0,
        answer_relevancy_score=0.95,
        context_precision_score=1.0,
        latency_ms=120.5,
        token_count=15,
        cost_usd=0.0003,
        passed=True,
        failure_reasons=[]
    )
    assert eval_res.query == "What is RAG?"
    assert eval_res.faithfulness_score == 1.0
    assert eval_res.latency_ms == 120.5
    assert eval_res.passed is True

    usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.0003)
    assert usage.total_tokens == 15


