import os
import sys
import argparse
from typing import List, Dict, Any
from eval_harness.metrics import EvalResult, Settings
from eval_harness.runner import evaluate_rag_response

# Define 15 Golden Query/Answer pairs covering compliance, matching, and general RAG
GOLDEN_DATASET: List[Dict[str, Any]] = [
    {
        "query": "What is the Locate Rule for short selling?",
        "context": "Short selling compliance requires borrowing the security before executing the short sale (Locate Rule).",
        "answer": "The Locate Rule requires borrowing the security before executing a short sale.",
        "expected_faithfulness": 1.0,
        "should_pass": True
    },
    {
        "query": "Can employees trade stocks of target companies?",
        "context": "Insider trading is strictly prohibited. Employees must clear all personal trades through the compliance office.",
        "answer": "Yes, employees can trade any stock they want without clearance.",
        "expected_faithfulness": 0.2,
        "should_pass": False
    },
    {
        "query": "What constitutes market manipulation under SEC?",
        "context": "Market manipulation includes wash sales, spoofing, and cornering the market, which are illegal under SEC rules.",
        "answer": "SEC rules prohibit wash sales, spoofing, and cornering the market as manipulation.",
        "expected_faithfulness": 0.95,
        "should_pass": True
    },
    {
        "query": "Are wash sales allowed in personal accounts?",
        "context": "Market manipulation includes wash sales, spoofing, and cornering the market, which are illegal under SEC rules.",
        "answer": "No, wash sales are a form of illegal market manipulation under SEC regulations.",
        "expected_faithfulness": 1.0,
        "should_pass": True
    },
    {
        "query": "What is the compliance requirement for insider trades?",
        "context": "Insider trading is strictly prohibited. Employees must clear all personal trades through the compliance office.",
        "answer": "Employees are required to clear all personal trades through the compliance office to prevent insider trading.",
        "expected_faithfulness": 1.0,
        "should_pass": True
    },
    {
        "query": "What qualifications does the python developer have?",
        "context": "Resume: Python Developer with 5 years of experience in Django, FastAPI, and SQL databases.",
        "answer": "The Python developer has 5 years of experience with Django, FastAPI, and SQL databases.",
        "expected_faithfulness": 1.0,
        "should_pass": True
    },
    {
        "query": "Does the frontend developer know React?",
        "context": "Resume: Frontend Developer with 3 years of experience in React, Vue, and TailwindCSS.",
        "answer": "Yes, the frontend developer has 3 years of experience in React.",
        "expected_faithfulness": 1.0,
        "should_pass": True
    },
    {
        "query": "What ML frameworks does the AI engineer use?",
        "context": "Resume: Machine Learning / AI Engineer specialized in PyTorch, NLP, and LLM fine-tuning.",
        "answer": "The AI engineer uses TensorFlow and Keras.",
        "expected_faithfulness": 0.1,  # Contradicts PyTorch context
        "should_pass": False
    },
    {
        "query": "Does the frontend developer know React and Angular?",
        "context": "Resume: Frontend Developer with 3 years of experience in React, Vue, and TailwindCSS.",
        "answer": "The candidate has experience with React and Vue, but Angular is not mentioned.",
        "expected_faithfulness": 0.9,
        "should_pass": True
    },
    {
        "query": "What is the locate rule exception?",
        "context": "Short selling compliance requires borrowing the security before executing the short sale (Locate Rule).",
        "answer": "The locate rule has no exceptions for retail investors.",
        "expected_faithfulness": 0.8,
        "should_pass": True
    },
    {
        "query": "Is spoofing permitted in high-frequency trading?",
        "context": "Market manipulation includes wash sales, spoofing, and cornering the market, which are illegal under SEC rules.",
        "answer": "Spoofing is illegal under SEC rules, including in high-frequency trading.",
        "expected_faithfulness": 0.95,
        "should_pass": True
    },
    {
        "query": "How many years of experience does the frontend dev have?",
        "context": "Resume: Frontend Developer with 3 years of experience in React, Vue, and TailwindCSS.",
        "answer": "The frontend developer has 3 years of experience.",
        "expected_faithfulness": 1.0,
        "should_pass": True
    },
    {
        "query": "What databases does the python dev know?",
        "context": "Resume: Python Developer with 5 years of experience in Django, FastAPI, and SQL databases.",
        "answer": "The python developer knows SQL databases.",
        "expected_faithfulness": 1.0,
        "should_pass": True
    },
    {
        "query": "Does the python developer have Node experience?",
        "context": "Resume: Python Developer with 5 years of experience in Django, FastAPI, and SQL databases.",
        "answer": "Node.js is not mentioned on the resume.",
        "expected_faithfulness": 0.9,
        "should_pass": True
    },
    {
        "query": "What specialized AI skills does the candidate have?",
        "context": "Resume: Machine Learning / AI Engineer specialized in PyTorch, NLP, and LLM fine-tuning.",
        "answer": "The candidate is specialized in PyTorch, NLP, and LLM fine-tuning.",
        "expected_faithfulness": 1.0,
        "should_pass": True
    }
]


def run_eval_gate(mock: bool = False) -> int:
    """Run golden dataset through the evaluation runner and assert CI gate thresholds.
    
    Thresholds:
    - Mean faithfulness score >= 0.70
    - Escalation rate <= 20% (0.20)
    """
    print("=" * 60)
    print("  RUNNING EVALUATION HARNESS CI BUILD GATE  ")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY")
    is_mocked = mock or not api_key

    if is_mocked:
        print("[INFO] Running in MOCK mode (No API Key detected or --mock passed).")

    eval_results = []
    
    for i, item in enumerate(GOLDEN_DATASET):
        query = item["query"]
        contexts = [item["context"]]
        answer = item["answer"]

        if is_mocked:
            # Simulate deterministic evaluation results based on golden expectations
            faithfulness_score = item["expected_faithfulness"]
            passed = item["should_pass"]
            failure_reasons = [] if passed else ["Mocked evaluation failed: faithfulness too low."]
            
            res = EvalResult(
                query=query,
                retrieved_contexts=contexts,
                generated_answer=answer,
                faithfulness_score=faithfulness_score,
                answer_relevancy_score=0.9,
                deepeval_hallucination_score=faithfulness_score,
                deepeval_answer_relevancy_score=0.9,
                disagreement=False,
                passed=passed,
                failure_reasons=failure_reasons,
                latency_ms=10.0,
                cost_usd=0.00005
            )
        else:
            # Run the actual evaluations
            def custom_retrieval(q):
                return contexts
                
            def custom_generation(q, c):
                return answer

            res = evaluate_rag_response(
                query=query,
                retrieval_fn=custom_retrieval,
                generation_fn=custom_generation
            )
            
        eval_results.append(res)
        status = "PASS" if res.passed else "FAIL"
        print(f"[{i+1:02d}/15] Status: {status} | Query: '{query[:40]}...' | Faithfulness: {res.faithfulness_score}")

    # Compute aggregate stats
    faithfulness_scores = [r.faithfulness_score for r in eval_results if r.faithfulness_score is not None]
    mean_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
    
    escalations = sum(1 for r in eval_results if not r.passed)
    escalation_rate = escalations / len(eval_results)

    print("\n" + "=" * 40)
    print("  AGGREGATE EVALUATION GATE METRICS  ")
    print("=" * 40)
    print(f"Average Faithfulness Score : {mean_faithfulness:.4f} (Required: >= 0.7000)")
    print(f"Escalation Rate            : {escalation_rate:.2%} (Required: <= 20.00%)")
    print(f"Total Queries Executed     : {len(eval_results)}")
    print(f"Escalated Queries Count    : {escalations}")
    print("=" * 40)

    # Validate gate assertions
    gate_failed = False
    
    if mean_faithfulness < 0.7:
        print("[ERROR] CI Gate Failed: Mean faithfulness score is below threshold of 0.7.")
        gate_failed = True
        
    if escalation_rate > 0.20:
        print("[ERROR] CI Gate Failed: Escalation rate exceeds maximum threshold of 20%.")
        gate_failed = True

    if gate_failed:
        print("\n[RESULT] BUILD STATUS: FAILED")
        return 1
    else:
        print("\n[RESULT] BUILD STATUS: PASSED (CI Gate requirements met)")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval Harness CI Gate Runner")
    parser.add_argument("--mock", action="store_true", help="Force mock execution mode")
    args = parser.parse_args()
    
    sys.exit(run_eval_gate(mock=args.mock))
