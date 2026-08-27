import os
import json
import time
from typing import Dict, Any, Callable, TypeVar, Optional, List
from functools import wraps
from eval_harness.metrics import EvalResult
from eval_harness.database import get_mongo_db

T = TypeVar('T')


def trace_pipeline_run(
    query: str,
    eval_result: EvalResult,
    route_decision: str,
    domain_description: Optional[str] = None
) -> None:
    """Log a structured trace of the RAG pipeline run.
    
    Persists to MongoDB if MONGO_URI is set. Also logs to Langfuse if keys are set.
    Falls back to a local JSON-lines file if no remote database or trace client is active.
    
    Args:
        query: The input query string.
        eval_result: The EvalResult containing scores, cost, and latency.
        route_decision: The routing decision ("serve", "caveat", "escalate").
        domain_description: Optional product domain description (e.g. trading compliance).
    """
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    trace_data = {
        "timestamp": time.time(),
        "query": query,
        "domain_description": domain_description or os.environ.get("RAG_DOMAIN_DESCRIPTION", "unknown"),
        "retrieved_contexts_count": len(eval_result.retrieved_contexts),
        "faithfulness_score": eval_result.faithfulness_score,
        "answer_relevancy_score": eval_result.answer_relevancy_score,
        "context_precision_score": eval_result.context_precision_score,
        "deepeval_hallucination_score": eval_result.deepeval_hallucination_score,
        "deepeval_answer_relevancy_score": eval_result.deepeval_answer_relevancy_score,
        "total_cost_usd": eval_result.cost_usd,
        "latency_ms": eval_result.latency_ms,
        "route_decision": route_decision,
        "passed": eval_result.passed,
        "failure_reasons": eval_result.failure_reasons,
        "agent_reasoning": eval_result.agent_reasoning
    }

    # 1. MongoDB Persistence
    db = get_mongo_db()
    written_to_db = False
    if db is not None:
        try:
            db.traces.insert_one(trace_data.copy())
            written_to_db = True
        except Exception:
            pass

    # 2. Langfuse Observability Tracing
    logged_to_langfuse = False
    if public_key and secret_key:
        try:
            from langfuse import Langfuse
            langfuse = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
            # Create Langfuse Trace
            trace = langfuse.trace(
                name="rag-pipeline-run",
                input={"query": query},
                output={"generated_answer": eval_result.generated_answer},
                metadata=trace_data
            )
            # Add a span representing the RAG evaluation harness
            trace.span(
                name="evaluation-harness",
                input=query,
                output=eval_result.model_dump(),
                metadata={"passed": eval_result.passed}
            )
            langfuse.flush()
            logged_to_langfuse = True
        except Exception:
            pass

    # 3. Fallback to Local JSONL File (if not persisted elsewhere)
    if not written_to_db and not logged_to_langfuse:
        log_dir = os.environ.get("EVAL_HARNESS_LOG_DIR", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "traces.jsonl")
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace_data) + "\n")


def trace_step(name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to trace a single step/function in the RAG pipeline using Langfuse."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # No-op trace step if Langfuse is not initialized or used dynamically
            return func(*args, **kwargs)
        return wrapper
    return decorator


class TraceManager:
    """Manages the lifetime of a trace block for Langfuse logging."""
    
    def __init__(self, host: Optional[str] = None, public_key: Optional[str] = None, secret_key: Optional[str] = None) -> None:
        self.host = host or os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self.public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
        self.client = None
        self.current_trace = None
        
        if self.public_key and self.secret_key:
            try:
                from langfuse import Langfuse
                self.client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host
                )
            except Exception:
                pass
        
    def create_trace(self, name: str, input_data: Any) -> None:
        """Initialize a new execution trace."""
        if self.client:
            try:
                self.current_trace = self.client.trace(name=name, input=input_data)
            except Exception:
                pass
        
    def end_trace(self, output_data: Any, status: str = "success") -> None:
        """End the current trace and log the final output."""
        if self.client and self.current_trace:
            try:
                self.current_trace.update(output=output_data, metadata={"status": status})
                self.client.flush()
            except Exception:
                pass
            finally:
                self.current_trace = None
