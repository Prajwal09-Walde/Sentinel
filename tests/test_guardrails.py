import pytest
from unittest.mock import patch, MagicMock
from eval_harness.metrics import EvalResult
from eval_harness.guardrails import (
    check_out_of_scope,
    check_hallucination,
    route_response
)

# ==========================================
# 1. Test check_out_of_scope
# ==========================================

@patch("eval_harness.guardrails.OpenAI")
@patch.dict("os.environ", {"OPENAI_API_KEY": "mock-key"})
def test_check_out_of_scope_in_scope(mock_openai_class):
    # Mock chat completions response to return IN_SCOPE
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    
    mock_message.content = "IN_SCOPE"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    out_of_scope = check_out_of_scope("What are my trading compliance rules?", "trading documentation")
    assert out_of_scope is False
    mock_client.chat.completions.create.assert_called_once()

@patch("eval_harness.guardrails.OpenAI")
@patch.dict("os.environ", {"OPENAI_API_KEY": "mock-key"})
def test_check_out_of_scope_out_of_scope(mock_openai_class):
    # Mock chat completions response to return OUT_OF_SCOPE
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    
    mock_message.content = "OUT_OF_SCOPE"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    out_of_scope = check_out_of_scope("How do I bake a chocolate cake?", "trading documentation")
    assert out_of_scope is True

@patch("eval_harness.guardrails.OpenAI")
@patch.dict("os.environ", {"OPENAI_API_KEY": "mock-key"})
def test_check_out_of_scope_exception(mock_openai_class):
    # Mock client to throw exception
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API connection timed out")
    mock_openai_class.return_value = mock_client

    # Fail-safe check: should return False (in-scope) on errors
    out_of_scope = check_out_of_scope("test query", "trading documentation")
    assert out_of_scope is False

@patch("eval_harness.guardrails.OpenAI", None)
def test_check_out_of_scope_no_client():
    # If API key is missing or OpenAI is None
    out_of_scope = check_out_of_scope("test query", "trading documentation")
    assert out_of_scope is False


# ==========================================
# 2. Test check_hallucination
# ==========================================

@patch("eval_harness.guardrails.HallucinationMetric")
@patch("eval_harness.guardrails.LLMTestCase")
@patch.dict("os.environ", {"DEEPEVAL_HALLUCINATION_THRESHOLD": "0.7"})
def test_check_hallucination_not_hallucinating(mock_test_case_cls, mock_metric_cls):
    mock_metric = MagicMock()
    mock_metric.score = 0.9  # High score = faithful
    mock_metric.reason = "The claim is fully supported by the context."
    mock_metric_cls.return_value = mock_metric

    is_hallucinating, explanation = check_hallucination("A blond drinking water.", ["A blond-haired man drinking water."])
    assert is_hallucinating is False
    assert explanation == "The claim is fully supported by the context."
    mock_metric.measure.assert_called_once()

@patch("eval_harness.guardrails.HallucinationMetric")
@patch("eval_harness.guardrails.LLMTestCase")
@patch.dict("os.environ", {"DEEPEVAL_HALLUCINATION_THRESHOLD": "0.7"})
def test_check_hallucination_is_hallucinating(mock_test_case_cls, mock_metric_cls):
    mock_metric = MagicMock()
    mock_metric.score = 0.4  # Low score = hallucinated
    mock_metric.reason = "The claim contradicts the context."
    mock_metric_cls.return_value = mock_metric

    is_hallucinating, explanation = check_hallucination("A blond drinking cola.", ["A blond-haired man drinking water."])
    assert is_hallucinating is True
    assert explanation == "The claim contradicts the context."

def test_check_hallucination_empty_contexts():
    # Verify behavior when context list is empty
    is_hallucinating, explanation = check_hallucination("test answer", [])
    assert is_hallucinating is True
    assert "No retrieved contexts provided" in explanation


# ==========================================
# 3. Test route_response
# ==========================================

def test_route_response_serve():
    # passed=True, disagreement=False -> serve
    eval_res = EvalResult(
        query="q", retrieved_contexts=["c"], generated_answer="a",
        passed=True, disagreement=False
    )
    assert route_response(eval_res) == "serve"

def test_route_response_caveat():
    # passed=True, disagreement=True -> caveat
    eval_res = EvalResult(
        query="q", retrieved_contexts=["c"], generated_answer="a",
        passed=True, disagreement=True
    )
    assert route_response(eval_res) == "caveat"

def test_route_response_escalate():
    # passed=False, disagreement=False -> escalate
    eval_res1 = EvalResult(
        query="q", retrieved_contexts=["c"], generated_answer="a",
        passed=False, disagreement=False
    )
    assert route_response(eval_res1) == "escalate"

    # passed=False, disagreement=True -> escalate
    eval_res2 = EvalResult(
        query="q", retrieved_contexts=["c"], generated_answer="a",
        passed=False, disagreement=True
    )
    assert route_response(eval_res2) == "escalate"
