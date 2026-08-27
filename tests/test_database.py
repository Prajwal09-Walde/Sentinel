import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from eval_harness.metrics import EvalResult
from eval_harness.tracing import trace_pipeline_run
from eval_harness.database import get_mongo_db
from trading_copilot.app import app as trading_app


def test_get_mongo_db_not_configured():
    # If MONGO_URI is missing, get_mongo_db must return None
    with patch.dict(os.environ, {"MONGO_URI": ""}):
        db = get_mongo_db()
        assert db is None


@patch("eval_harness.database.MongoClient")
def test_get_mongo_db_ping_failure(mock_client_cls):
    # If connection fails ping checks, get_mongo_db must return None
    mock_client = MagicMock()
    mock_client.admin.command.side_effect = Exception("Connection Refused")
    mock_client_cls.return_value = mock_client

    with patch.dict(os.environ, {"MONGO_URI": "mongodb://test:27017"}):
        db = get_mongo_db()
        assert db is None


@patch("eval_harness.tracing.get_mongo_db")
def test_trace_pipeline_run_to_mongodb(mock_get_db):
    # Mock MongoDB connection and traces collection
    mock_db = MagicMock()
    mock_get_db.return_value = mock_db
    
    eval_res = EvalResult(
        query="Test query",
        retrieved_contexts=["Context 1"],
        generated_answer="Answer 1",
        passed=True,
        cost_usd=0.0001,
        latency_ms=100.0,
        faithfulness_score=0.9,
        answer_relevancy_score=0.95
    )

    with patch("eval_harness.tracing.time.time", return_value=123456.78):
        trace_pipeline_run(
            query="Test query",
            eval_result=eval_res,
            route_decision="serve",
            domain_description="trading compliance"
        )

    # Verify MongoDB insert_one was called with correctly structured trace document
    mock_db.traces.insert_one.assert_called_once()
    trace_data = mock_db.traces.insert_one.call_args[0][0]
    
    assert trace_data["query"] == "Test query"
    assert trace_data["domain_description"] == "trading compliance"
    assert trace_data["timestamp"] == 123456.78
    assert trace_data["total_cost_usd"] == 0.0001
    assert trace_data["faithfulness_score"] == 0.9


@patch("trading_copilot.app.get_mongo_db")
def test_eval_metrics_query_from_mongodb(mock_get_db):
    # Mock MongoDB metrics retrieval database query
    mock_db = MagicMock()
    mock_get_db.return_value = mock_db

    # Prepopulate mock cursor results
    mock_db.traces.find.return_value.sort.return_value.limit.return_value = [
        {
            "_id": "60d5ec49c63c3f3050b1c2b1",
            "query": "query 1",
            "faithfulness_score": 0.8,
            "answer_relevancy_score": 0.85,
            "total_cost_usd": 0.001,
            "route_decision": "serve"
        },
        {
            "_id": "60d5ec49c63c3f3050b1c2b2",
            "query": "query 2",
            "faithfulness_score": 0.4,
            "answer_relevancy_score": 0.5,
            "total_cost_usd": 0.002,
            "route_decision": "escalate"
        }
    ]

    client = TestClient(trading_app)
    response = client.get("/eval-metrics?n=10&format=json")
    
    assert response.status_code == 200
    data = response.json()
    
    # Assert MongoDB metrics aggregation math
    assert data["query_count"] == 2
    assert pytest.approx(data["mean_faithfulness"], abs=1e-4) == 0.6
    assert pytest.approx(data["mean_relevancy"], abs=1e-4) == 0.675
    assert pytest.approx(data["escalation_rate"], abs=1e-4) == 0.5
    assert pytest.approx(data["average_cost"], abs=1e-6) == 0.0015


def test_generate_agent_reasoning():
    from eval_harness.runner import generate_agent_reasoning
    eval_res = EvalResult(
        query="Test query",
        retrieved_contexts=["Context 1"],
        generated_answer="Answer 1",
        passed=True,
        faithfulness_score=0.9,
        answer_relevancy_score=0.85
    )
    
    # Assert offline programmatic fallback justifications
    reason = generate_agent_reasoning(eval_res, "serve")
    assert "Both RAGAS and DeepEval metrics passed safety checks" in reason
    
    eval_res.failure_reasons = ["Score below threshold"]
    reason_esc = generate_agent_reasoning(eval_res, "escalate")
    assert "Response failed safety or hallucination checks" in reason_esc

