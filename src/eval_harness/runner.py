import os
import time
from typing import Dict, Any, List, Callable, Optional, Tuple
from eval_harness.metrics import EvalResult, Settings, run_dual_eval
from eval_harness.guardrails import check_out_of_scope, check_hallucination, route_response
from eval_harness.cost_tracker import CostTracker
from eval_harness.tracing import trace_pipeline_run


def generate_agent_reasoning(eval_result: EvalResult, route_decision: str) -> str:
    """Generate agent reasoning justifying the routing decision based on evaluation metrics."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            prompt = (
                f"You are a RAG Router Agent. Justify in under 15 words why this response was routed to '{route_decision.upper()}'.\n"
                f"Scores:\n"
                f"- Faithfulness: {eval_result.faithfulness_score}\n"
                f"- Relevancy: {eval_result.answer_relevancy_score}\n"
                f"- Hallucination check failed: {not eval_result.passed and 'Yes' or 'No'}\n"
                f"- Framework conflict: {eval_result.disagreement}\n"
                f"Respond with ONLY the 1-sentence justification."
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=30,
                temperature=0.0
            )
            reason = response.choices[0].message.content.strip()
            return reason.replace('"', '').replace("'", "")
        except Exception:
            pass

    # Programmatic fallback reasoning
    if route_decision == "serve":
        f_score = eval_result.faithfulness_score
        f_str = f" ({f_score:.2f})" if f_score is not None else ""
        return f"Approved: Both RAGAS and DeepEval metrics passed safety checks{f_str}."
    elif route_decision == "caveat":
        return "Warning: Framework disagreement between Ragas and DeepEval metrics."
    else:
        # Escalate
        reasons = ", ".join(eval_result.failure_reasons)
        if "out of scope" in reasons.lower():
            return "Rejected: Query is out of scope for the compliance domain."
        return "Blocked: Response failed safety or hallucination checks."


def evaluate_rag_response(
    query: str,
    retrieval_fn: Callable[[str], List[str]],
    generation_fn: Callable[[str, List[str]], str],
    domain_description: Optional[str] = None,
    settings: Optional[Settings] = None
) -> EvalResult:
    """Orchestrates the entire RAG pipeline execution, evaluation, and guardrail lifecycle.
    
    Order of operations:
    1. Out-of-scope domain check (runs before retrieval to save cost)
    2. Retrieval & Generation (wrapped in CostTracker)
    3. Dual metrics evaluation (RAGAS + DeepEval)
    4. Hallucination check guardrail
    5. Cost and latency accounting
    6. Response routing decision ("serve", "caveat", "escalate")
    7. Production tracing log (Langfuse or local fallback)
    
    Args:
        query: The input user query.
        retrieval_fn: Callable that takes query -> list of retrieved contexts.
        generation_fn: Callable that takes query, contexts -> generated answer.
        domain_description: Optional description of the allowed system domain.
        settings: Optional custom evaluation thresholds.
        
    Returns:
        EvalResult object representing the full pipeline execution status and metrics.
    """
    if settings is None:
        settings = Settings()

    # 1. Out-of-scope Domain Check (executed before retrieval)
    if domain_description is None:
        domain_description = os.environ.get("RAG_DOMAIN_DESCRIPTION")

    if domain_description:
        if check_out_of_scope(query, domain_description):
            eval_result = EvalResult(
                query=query,
                retrieved_contexts=[],
                generated_answer="I am sorry, but your query is outside my declared domain.",
                passed=False,
                failure_reasons=["Query is out of scope for the declared domain."],
                agent_reasoning="Rejected: Query is out of scope for the declared domain."
            )
            # Log out-of-scope early refuse trace
            trace_pipeline_run(query, eval_result, "escalate", domain_description=domain_description)
            return eval_result

    # 2. Wrapped request lifecycle with CostTracker
    start_time = time.perf_counter()
    with CostTracker() as ct:
        try:
            # Execute Retrieval
            retrieved_contexts = retrieval_fn(query)
            
            # Execute Generation
            generated_answer = generation_fn(query, retrieved_contexts)
            
            # 3. Dual Metrics Evaluation (RAGAS + DeepEval)
            eval_result = run_dual_eval(
                query=query,
                retrieved_contexts=retrieved_contexts,
                generated_answer=generated_answer,
                settings=settings
            )
            
            # 4. Hallucination Guardrail Check
            is_hallucinating, hallucination_reason = check_hallucination(
                generated_answer,
                retrieved_contexts
            )
            if is_hallucinating:
                eval_result.passed = False
                eval_result.failure_reasons.append(f"Hallucination check failed: {hallucination_reason}")
                
        except Exception as e:
            # Handle RAG runtime errors gracefully
            eval_result = EvalResult(
                query=query,
                retrieved_contexts=[],
                generated_answer="An error occurred while processing your request.",
                passed=False,
                failure_reasons=[f"RAG pipeline execution error: {str(e)}"]
            )

        # 5. Cost and Latency Accounting
        summary = ct.get_summary()
        eval_result.token_count = summary["total_tokens"]
        eval_result.cost_usd = summary["total_cost_usd"]
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        eval_result.latency_ms = elapsed_ms

        # 6. Response Routing Decision
        route_decision = route_response(eval_result)
        
        if route_decision == "caveat":
            eval_result.generated_answer += (
                "\n\n*(Note: Evaluation frameworks disagreed on this response's consistency. "
                "Please verify details.)*"
            )
        elif route_decision == "escalate":
            eval_result.generated_answer = (
                "I apologize, but this response failed our internal consistency checks. "
                "A support ticket has been logged."
            )

        # 6.5 AI Routing Agent Reasoning
        eval_result.agent_reasoning = generate_agent_reasoning(eval_result, route_decision)

        # 7. Obs/Tracing Logs
        trace_pipeline_run(query, eval_result, route_decision, domain_description=domain_description)

        return eval_result


class EvalRunner:
    """Runner wrapper to support original package structure requirements."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.domain_description = config.get("domain_description")
        self.settings = config.get("settings")

    def run_eval(self, rag_func: Callable[[str], Dict[str, Any]], dataset: List[Dict[str, Any]]) -> List[EvalResult]:
        """Run evaluation on a dataset using the provided RAG function/pipeline."""
        results = []
        for item in dataset:
            query = item["query"]
            # Adapt the RAG callable to retrieval/generation functions
            def mock_retrieval(q: str) -> List[str]:
                res = rag_func(q)
                return res.get("contexts", [])
            
            def mock_generation(q: str, c: List[str]) -> str:
                res = rag_func(q)
                return res.get("response", "")

            eval_res = evaluate_rag_response(
                query=query,
                retrieval_fn=mock_retrieval,
                generation_fn=mock_generation,
                domain_description=self.domain_description,
                settings=self.settings
            )
            results.append(eval_res)
        return results
