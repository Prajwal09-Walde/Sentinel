import os
import json
import pytest
from unittest.mock import patch, MagicMock
from eval_harness.metrics import EvalResult, Settings
from eval_harness.runner import evaluate_rag_response, EvalRunner

@pytest.fixture
def mock_rag_fns():
    retrieval_fn = MagicMock(return_value=["retrieved context paragraph"])
    generation_fn = MagicMock(return_value="The generated RAG answer.")
    return retrieval_fn, generation_fn


@patch("eval_harness.runner.check_out_of_scope")
@patch("eval_harness.runner.trace_pipeline_run")
def test_evaluate_rag_response_out_of_scope(mock_trace, mock_scope_check, mock_rag_fns, tmpdir):
    retrieval_fn, generation_fn = mock_rag_fns
    mock_scope_check.return_value = True

    # Run the evaluation
    res = evaluate_rag_response(
        query="Out of domain prompt?",
        retrieval_fn=retrieval_fn,
        generation_fn=generation_fn,
        domain_description="trading compliance"
    )

    # Assertions for early out-of-scope refusal
    assert res.passed is False
    assert "Query is out of scope" in res.failure_reasons[0]
    assert res.generated_answer == "I am sorry, but your query is outside my declared domain."
    assert res.retrieved_contexts == []
    
    # Ensure retrieval and generation functions were NOT called
    retrieval_fn.assert_not_called()
    generation_fn.assert_not_called()
    
    # Ensure trace was logged
    mock_trace.assert_called_once()
    assert mock_trace.call_args[0][2] == "escalate"


@patch("eval_harness.runner.check_out_of_scope")
@patch("eval_harness.runner.run_dual_eval")
@patch("eval_harness.runner.check_hallucination")
@patch("eval_harness.runner.CostTracker")
def test_evaluate_rag_response_normal_flow(
    mock_cost_tracker_cls,
    mock_hallucination_check,
    mock_dual_eval,
    mock_scope_check,
    mock_rag_fns,
    tmpdir
):
    retrieval_fn, generation_fn = mock_rag_fns
    
    # 1. Setup out-of-scope check to pass (False)
    mock_scope_check.return_value = False

    # 2. Setup dual eval to pass
    mock_eval_res = EvalResult(
        query="In domain prompt",
        retrieved_contexts=["retrieved context paragraph"],
        generated_answer="The generated RAG answer.",
        passed=True,
        disagreement=False,
        faithfulness_score=0.9,
        answer_relevancy_score=0.9
    )
    mock_dual_eval.return_value = mock_eval_res

    # 3. Setup hallucination check to pass
    mock_hallucination_check.return_value = (False, "Supported claims")

    # 4. Setup mock cost tracker to yield custom usage
    mock_ct = MagicMock()
    mock_ct.get_summary.return_value = {
        "total_tokens": 1500,
        "total_cost_usd": 0.00225
    }
    # Make context manager return the mock instance
    mock_cost_tracker_cls.return_value.__enter__.return_value = mock_ct

    # Configure temp folder for trace file logging
    log_dir = str(tmpdir.mkdir("logs"))
    with patch.dict(os.environ, {"EVAL_HARNESS_LOG_DIR": log_dir, "LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": ""}):
        res = evaluate_rag_response(
            query="In domain prompt",
            retrieval_fn=retrieval_fn,
            generation_fn=generation_fn,
            domain_description="trading compliance"
        )

    # Assertions
    assert res.passed is True
    assert res.generated_answer == "The generated RAG answer."
    assert res.token_count == 1500
    assert res.cost_usd == 0.00225
    assert res.latency_ms > 0.0

    retrieval_fn.assert_called_once_with("In domain prompt")
    generation_fn.assert_called_once_with("In domain prompt", ["retrieved context paragraph"])
    
    # Verify local fallback trace file was written
    trace_file_path = os.path.join(log_dir, "traces.jsonl")
    assert os.path.exists(trace_file_path)
    
    with open(trace_file_path, "r", encoding="utf-8") as f:
        traces = [json.loads(line) for line in f]
        
    assert len(traces) == 1
    assert traces[0]["query"] == "In domain prompt"
    assert traces[0]["route_decision"] == "serve"
    assert traces[0]["total_cost_usd"] == 0.00225


@patch("eval_harness.runner.check_out_of_scope")
@patch("eval_harness.runner.run_dual_eval")
@patch("eval_harness.runner.check_hallucination")
@patch("eval_harness.runner.CostTracker")
def test_evaluate_rag_response_caveat_routing(
    mock_cost_tracker_cls,
    mock_hallucination_check,
    mock_dual_eval,
    mock_scope_check,
    mock_rag_fns,
    tmpdir
):
    retrieval_fn, generation_fn = mock_rag_fns
    mock_scope_check.return_value = False

    # Setup dual eval to have passed=True but disagreement=True
    mock_eval_res = EvalResult(
        query="Prompt",
        retrieved_contexts=["Context"],
        generated_answer="Answer.",
        passed=True,
        disagreement=True,
        faithfulness_score=0.8,
        answer_relevancy_score=0.8
    )
    mock_dual_eval.return_value = mock_eval_res
    mock_hallucination_check.return_value = (False, "Supported claims")

    # Mock CostTracker
    mock_ct = MagicMock()
    mock_ct.get_summary.return_value = {"total_tokens": 100, "total_cost_usd": 0.0}
    mock_cost_tracker_cls.return_value.__enter__.return_value = mock_ct

    log_dir = str(tmpdir.mkdir("logs"))
    with patch.dict(os.environ, {"EVAL_HARNESS_LOG_DIR": log_dir}):
        res = evaluate_rag_response(
            query="Prompt",
            retrieval_fn=retrieval_fn,
            generation_fn=generation_fn
        )

    # Decision should be "caveat", appending caveat warning to output answer
    assert res.passed is True
    assert "Evaluation frameworks disagreed on this response's consistency" in res.generated_answer


def test_eval_runner_dataset_interface():
    # Verify dataset EvalRunner interface wrapper works correctly
    runner = EvalRunner(config={"domain_description": "compliance"})
    
    mock_rag = MagicMock(return_value={"contexts": ["c"], "response": "a"})

    with patch("eval_harness.runner.evaluate_rag_response") as mock_eval_fn:
        mock_eval_fn.return_value = EvalResult(
            query="q", retrieved_contexts=["c"], generated_answer="a", passed=True
        )
        
        results = runner.run_eval(mock_rag, [{"query": "q"}])
        
        assert len(results) == 1
        assert results[0].passed is True
        mock_eval_fn.assert_called_once()
