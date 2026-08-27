import os
import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import both applications
from trading_copilot.app import app as trading_app
from talentiq.app import app as talentiq_app

@pytest.fixture
def temp_log_dir(tmpdir):
    return str(tmpdir.mkdir("logs"))


# ==========================================
# 1. Test Trading Copilot
# ==========================================

@patch("trading_copilot.app.evaluate_rag_response")
def test_trading_copilot_query_success(mock_eval_response):
    # Setup mock evaluate_rag_response to return a valid EvalResult
    mock_res = MagicMock()
    mock_res.generated_answer = "Based on our trading rules: Borrow security before short selling."
    mock_res.passed = True
    mock_res.latency_ms = 150.0
    mock_res.cost_usd = 0.0001
    mock_eval_response.return_value = mock_res

    client = TestClient(trading_app)
    response = client.post("/query", json={"query": "short selling rules"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["passed"] is True
    assert data["escalated"] is False
    assert "Borrow security before short selling" in data["response"]
    mock_eval_response.assert_called_once()

@patch("trading_copilot.app.evaluate_rag_response")
def test_trading_copilot_query_escalated(mock_eval_response):
    mock_res = MagicMock()
    mock_res.generated_answer = "I apologize, but this response failed our internal consistency checks."
    mock_res.passed = False
    mock_res.latency_ms = 200.0
    mock_res.cost_usd = 0.0003
    mock_eval_response.return_value = mock_res

    client = TestClient(trading_app)
    response = client.post("/query", json={"query": "out of domain prompt"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["passed"] is False
    assert data["escalated"] is True
    assert "failed our internal consistency checks" in data["response"]


# ==========================================
# 2. Test TalentIQ Copilot
# ==========================================

@patch("talentiq.app.evaluate_rag_response")
def test_talentiq_copilot_query_success(mock_eval_response):
    mock_res = MagicMock()
    mock_res.generated_answer = "Matching candidates found: Python Developer with 5 years exp."
    mock_res.passed = True
    mock_res.latency_ms = 120.0
    mock_res.cost_usd = 0.0001
    mock_eval_response.return_value = mock_res

    client = TestClient(talentiq_app)
    response = client.post("/query", json={"query": "need python developer"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["passed"] is True
    assert "Python Developer" in data["response"]


# ==========================================
# 3. Test /eval-metrics Aggregation
# ==========================================

def test_eval_metrics_empty(temp_log_dir):
    client = TestClient(trading_app)
    with patch.dict(os.environ, {"EVAL_HARNESS_LOG_DIR": temp_log_dir}):
        response = client.get("/eval-metrics?format=json")
        
    assert response.status_code == 200
    data = response.json()
    assert data["query_count"] == 0
    assert data["mean_faithfulness"] == 0.0

def test_eval_metrics_aggregation(temp_log_dir):
    # Setup mock trace logs
    trace_file_path = os.path.join(temp_log_dir, "traces.jsonl")
    mock_traces = [
        {
            "query": "query 1",
            "faithfulness_score": 0.8,
            "answer_relevancy_score": 0.9,
            "total_cost_usd": 0.001,
            "route_decision": "serve"
        },
        {
            "query": "query 2",
            "faithfulness_score": 0.4,
            "answer_relevancy_score": 0.5,
            "total_cost_usd": 0.002,
            "route_decision": "escalate"
        },
        {
            "query": "query 3",
            "faithfulness_score": 0.9,
            "answer_relevancy_score": 0.8,
            "total_cost_usd": 0.003,
            "route_decision": "serve"
        }
    ]
    
    with open(trace_file_path, "w", encoding="utf-8") as f:
        for t in mock_traces:
            f.write(json.dumps(t) + "\n")

    client = TestClient(trading_app)
    with patch.dict(os.environ, {"EVAL_HARNESS_LOG_DIR": temp_log_dir}):
        response = client.get("/eval-metrics?n=10&format=json")

    assert response.status_code == 200
    data = response.json()
    assert data["query_count"] == 3
    
    # Math check:
    # Faithfulness = (0.8 + 0.4 + 0.9) / 3 = 2.1 / 3 = 0.7
    # Relevancy = (0.9 + 0.5 + 0.8) / 3 = 2.2 / 3 = 0.7333
    # Escalation rate = 1 / 3 = 0.3333
    # Average cost = (0.001 + 0.002 + 0.003) / 3 = 0.002
    assert pytest.approx(data["mean_faithfulness"], abs=1e-4) == 0.7
    assert pytest.approx(data["mean_relevancy"], abs=1e-4) == 0.7333
    assert pytest.approx(data["escalation_rate"], abs=1e-4) == 0.3333
    assert pytest.approx(data["average_cost"], abs=1e-6) == 0.002
