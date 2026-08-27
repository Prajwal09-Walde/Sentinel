import pytest
from unittest.mock import patch, MagicMock
from scripts.ci_eval_gate import run_eval_gate


def test_ci_gate_pass():
    # Run gate in mock mode. Golden dataset results should pass
    exit_code = run_eval_gate(mock=True)
    assert exit_code == 0


@patch("scripts.ci_eval_gate.GOLDEN_DATASET")
def test_ci_gate_fail_faithfulness(mock_golden):
    # Mock golden dataset to return low faithfulness scores
    mock_golden.copy.return_value = []
    mock_golden.__iter__.return_value = [
        {
            "query": "Q1",
            "context": "C1",
            "answer": "A1",
            "expected_faithfulness": 0.3,  # Fail
            "should_pass": False
        }
    ]
    mock_golden.__len__.return_value = 1

    exit_code = run_eval_gate(mock=True)
    assert exit_code == 1  # Should fail build due to low faithfulness


@patch("scripts.ci_eval_gate.GOLDEN_DATASET")
def test_ci_gate_fail_escalation_rate(mock_golden):
    # Mock golden dataset to trigger high escalation rate (>20%)
    mock_golden.copy.return_value = []
    mock_golden.__iter__.return_value = [
        {
            "query": "Q1", "context": "C1", "answer": "A1",
            "expected_faithfulness": 1.0, "should_pass": False  # Escalated (100% escalation rate)
        }
    ]
    mock_golden.__len__.return_value = 1

    exit_code = run_eval_gate(mock=True)
    assert exit_code == 1  # Should fail build due to high escalation rate
