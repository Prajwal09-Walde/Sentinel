import pytest
from unittest.mock import patch, MagicMock
from eval_harness.metrics import (
    Settings,
    EvalResult,
    run_ragas_eval,
    run_deepeval_eval,
    run_dual_eval
)

@pytest.fixture
def mock_dataset():
    with patch("eval_harness.metrics.Dataset") as mock_ds:
        yield mock_ds

def test_run_ragas_eval_pass(mock_dataset):
    # Mock ragas.evaluate to return successful scores
    with patch("eval_harness.metrics.evaluate") as mock_evaluate:
        mock_evaluate.return_value = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.90,
            "context_precision": 0.95
        }
        
        settings = Settings(ragas_faithfulness_threshold=0.7, ragas_answer_relevancy_threshold=0.7)
        res = run_ragas_eval(
            query="test query",
            retrieved_contexts=["test context"],
            generated_answer="test answer",
            ground_truth="test ground truth",
            settings=settings
        )
        
        assert res.passed is True
        assert res.faithfulness_score == 0.85
        assert res.answer_relevancy_score == 0.90
        assert res.context_precision_score == 0.95
        assert len(res.failure_reasons) == 0

def test_run_ragas_eval_fail_threshold(mock_dataset):
    # Mock ragas.evaluate to return failing scores
    with patch("eval_harness.metrics.evaluate") as mock_evaluate:
        mock_evaluate.return_value = {
            "faithfulness": 0.50,
            "answer_relevancy": 0.80
        }
        
        settings = Settings(ragas_faithfulness_threshold=0.7, ragas_answer_relevancy_threshold=0.7)
        res = run_ragas_eval(
            query="test query",
            retrieved_contexts=["test context"],
            generated_answer="test answer",
            settings=settings
        )
        
        assert res.passed is False
        assert res.faithfulness_score == 0.50
        assert "faithfulness score 0.5000 is below threshold 0.7000" in res.failure_reasons[0]

def test_run_ragas_eval_nan_handling(mock_dataset):
    # Mock ragas.evaluate to return NaN values
    with patch("eval_harness.metrics.evaluate") as mock_evaluate:
        mock_evaluate.return_value = {
            "faithfulness": float("nan"),
            "answer_relevancy": 0.90
        }
        
        res = run_ragas_eval(
            query="test query",
            retrieved_contexts=["test context"],
            generated_answer="test answer"
        )
        
        assert res.passed is False
        assert res.faithfulness_score is None
        assert any("faithfulness score is NaN or missing" in r for r in res.failure_reasons)

def test_run_deepeval_eval_pass():
    # Mock DeepEval metrics
    with patch("eval_harness.metrics.HallucinationMetric") as mock_hallucination_cls, \
         patch("eval_harness.metrics.AnswerRelevancyMetric") as mock_relevancy_cls, \
         patch("eval_harness.metrics.LLMTestCase"):
        
        # Setup mock instances
        mock_hallucination = MagicMock()
        mock_hallucination.score = 0.9  # High score = good/faithful in DeepEval v4
        mock_hallucination_cls.return_value = mock_hallucination

        mock_relevancy = MagicMock()
        mock_relevancy.score = 0.85  # High relevancy = good
        mock_relevancy_cls.return_value = mock_relevancy

        settings = Settings(deepeval_hallucination_threshold=0.7, deepeval_answer_relevancy_threshold=0.7)
        res = run_deepeval_eval(
            query="test query",
            retrieved_contexts=["test context"],
            generated_answer="test answer",
            settings=settings
        )

        assert res.passed is True
        assert res.deepeval_hallucination_score == 0.9
        assert res.deepeval_answer_relevancy_score == 0.85
        assert len(res.failure_reasons) == 0

def test_run_deepeval_eval_fail():
    with patch("eval_harness.metrics.HallucinationMetric") as mock_hallucination_cls, \
         patch("eval_harness.metrics.AnswerRelevancyMetric") as mock_relevancy_cls, \
         patch("eval_harness.metrics.LLMTestCase"):
        
        # Setup mock instances
        mock_hallucination = MagicMock()
        mock_hallucination.score = 0.4  # Below threshold (0.7)
        mock_hallucination_cls.return_value = mock_hallucination

        mock_relevancy = MagicMock()
        mock_relevancy.score = 0.85
        mock_relevancy_cls.return_value = mock_relevancy

        settings = Settings(deepeval_hallucination_threshold=0.7, deepeval_answer_relevancy_threshold=0.7)
        res = run_deepeval_eval(
            query="test query",
            retrieved_contexts=["test context"],
            generated_answer="test answer",
            settings=settings
        )

        assert res.passed is False
        assert any("hallucination score 0.4000 is below threshold" in r for r in res.failure_reasons)

def test_run_dual_eval_agreement_pass(mock_dataset):
    with patch("eval_harness.metrics.evaluate") as mock_evaluate, \
         patch("eval_harness.metrics.HallucinationMetric") as mock_hallucination_cls, \
         patch("eval_harness.metrics.AnswerRelevancyMetric") as mock_relevancy_cls, \
         patch("eval_harness.metrics.LLMTestCase"):
        
        # Ragas passes
        mock_evaluate.return_value = {"faithfulness": 0.90, "answer_relevancy": 0.90}
        
        # DeepEval passes
        mock_hall = MagicMock()
        mock_hall.score = 0.9
        mock_hallucination_cls.return_value = mock_hall

        mock_rel = MagicMock()
        mock_rel.score = 0.85
        mock_relevancy_cls.return_value = mock_rel

        res = run_dual_eval(
            query="test query",
            retrieved_contexts=["test context"],
            generated_answer="test answer"
        )

        assert res.passed is True
        assert res.disagreement is False
        assert len(res.failure_reasons) == 0

def test_run_dual_eval_disagreement(mock_dataset):
    with patch("eval_harness.metrics.evaluate") as mock_evaluate, \
         patch("eval_harness.metrics.HallucinationMetric") as mock_hallucination_cls, \
         patch("eval_harness.metrics.AnswerRelevancyMetric") as mock_relevancy_cls, \
         patch("eval_harness.metrics.LLMTestCase"):
        
        # Ragas passes (faithfulness=0.8, relevancy=0.8)
        mock_evaluate.return_value = {"faithfulness": 0.80, "answer_relevancy": 0.80}
        
        # DeepEval fails (low hallucination score 0.2 < threshold 0.7)
        mock_hall = MagicMock()
        mock_hall.score = 0.2
        mock_hallucination_cls.return_value = mock_hall

        mock_rel = MagicMock()
        mock_rel.score = 0.85
        mock_relevancy_cls.return_value = mock_rel

        res = run_dual_eval(
            query="test query",
            retrieved_contexts=["test context"],
            generated_answer="test answer"
        )

        assert res.passed is False
        assert res.disagreement is True
        assert any("Framework disagreement detected" in r for r in res.failure_reasons)
