import os
import json
import time
import random
from dotenv import load_dotenv
from eval_harness.database import get_mongo_db

# Load env file to get database credentials
load_dotenv()

TRADING_QUERIES = [
    ("What is the Locate Rule for short selling?", "compliance requires borrowing the security before short selling", "You must locate and borrow securities before executing a short sale."),
    ("Are wash sales permitted?", "market manipulation includes wash sales, spoofing, which are illegal", "Wash sales are illegal wash trading activities and are strictly prohibited."),
    ("How do employees clear personal trades?", "Employees must clear all personal trades through the compliance office", "All employee stock trades must receive prior clearance from compliance."),
    ("What is spoofing in trading?", "spoofing is a form of market manipulation where orders are placed and canceled", "Spoofing is an illegal trading practice of creating fake market orders to trick other traders."),
    ("Is insider trading allowed?", "Insider trading is strictly prohibited", "No, trading stocks using material non-public information is illegal insider trading."),
    ("What is frontrunning?", "Frontrunning is trading on advance information of block trades, which is illegal", "Frontrunning is entering trade orders ahead of client block trades, which is prohibited."),
    ("Can compliance block a trade?", "Compliance offices have authority to block any employee personal trades", "Yes, the compliance department has full authority to block suspicious or restricted trades."),
    ("What are restricted securities?", "Restricted securities are those on the firm's ban list due to advisory deals", "Restricted securities are assets that employees are barred from trading due to active firm transactions."),
    ("What is the penalty for spoofing?", "SEC penalties for market manipulation include fines and trading bans", "Spoofing penalties include heavy SEC fines, disgorgement, and potential criminal charges."),
    ("How long must trading records be kept?", "Firms must preserve trading logs and communications for at least 6 years", "Trading records must be archived and kept secure for a minimum period of 6 years.")
]

TALENT_QUERIES = [
    ("Find a Senior Python Developer", "Resume: Python Developer with 5 years experience in Django and FastAPI", "Candidate matched: Senior Python developer with 5 years experience in Django/FastAPI."),
    ("Looking for a Frontend Engineer", "Resume: Frontend Developer with 3 years experience in React and Vue", "Candidate matched: React/Vue frontend specialist with 3 years experience."),
    ("Need an AI Researcher", "Resume: AI Engineer specialized in PyTorch, NLP, and LLMs", "Candidate matched: PyTorch AI engineer with specialized LLM training experience."),
    ("Do we have mobile developers?", "Resume: iOS developer with Swift experience", "Candidate matched: Swift mobile engineer with iOS production experience."),
    ("Search for a database administrator", "Resume: DBA with 8 years experience in PostgreSQL and Oracle", "Candidate matched: Senior DBA expert in Postgres and Oracle systems."),
    ("Find a DevOps candidate", "Resume: DevOps engineer with Kubernetes and AWS experience", "Candidate matched: DevOps engineer with Kubernetes containerization and AWS experience."),
    ("Looking for a Product Manager", "Resume: Product Manager with experience in Agile and Scrum", "Candidate matched: Agile product lead with Scrum certification and SaaS delivery experience."),
    ("Need a QA Analyst", "Resume: QA engineer with Selenium test automation experience", "Candidate matched: QA Automation engineer with Selenium and CI/CD testing skills."),
    ("Search for data scientist", "Resume: Data scientist with Pandas, Scikit-Learn, and SQL experience", "Candidate matched: Data scientist with robust ML modeling skills in python/Pandas."),
    ("Find a cybersecurity specialist", "Resume: Cybersecurity analyst with CISSP and network audit experience", "Candidate matched: Certified security specialist with CISSP credentials.")
]


def generate_mock_traces(count: int = 500):
    traces = []
    now = time.time()
    
    # Generate traces spanning the last 7 days
    for i in range(count):
        # Evenly split between trading compliance and recruitment matching domains
        is_trading = (i % 2 == 0)
        domain = "trading documentation and compliance regulations" if is_trading else "recruitment matching and resume screening"
        query_pool = TRADING_QUERIES if is_trading else TALENT_QUERIES
        
        query, context, answer = random.choice(query_pool)
        
        # Stagger timestamps backward over 7 days
        timestamp = now - (count - i) * (7 * 86400 / count)
        
        # Randomize score outcomes: 85% passed, 15% failed/hallucinated
        passed = random.random() > 0.15
        
        if passed:
            faithfulness = random.uniform(0.72, 1.0)
            relevancy = random.uniform(0.75, 1.0)
            disagreement = random.random() > 0.93  # 7% chance of framework disagreement
            route_decision = "caveat" if disagreement else "serve"
            failure_reasons = []
        else:
            faithfulness = random.uniform(0.2, 0.65)
            relevancy = random.uniform(0.4, 0.68)
            disagreement = random.random() > 0.7  # 30% disagreement on failures
            route_decision = "escalate"
            failure_reasons = ["Evaluation score is below safety thresholds."]

        # Calculate mock token usage and cost
        latency = random.uniform(250.0, 1800.0) if passed else random.uniform(50.0, 300.0) # out of scope/failures fail fast
        prompt_tokens = random.randint(800, 1500)
        completion_tokens = random.randint(150, 450)
        total_tokens = prompt_tokens + completion_tokens
        
        # Price matches gpt-4o-mini ($0.150 / 1M input, $0.600 / 1M output)
        cost = (prompt_tokens * 0.15 / 1_000_000) + (completion_tokens * 0.60 / 1_000_000)

        # Define mock agent reasoning
        if route_decision == "serve":
            reason = f"Approved: Both RAGAS and DeepEval metrics passed safety checks ({faithfulness:.2f})."
        elif route_decision == "caveat":
            reason = "Warning: Framework disagreement between Ragas and DeepEval metrics."
        else:
            reason = "Blocked: Response failed safety or hallucination checks."

        trace = {
            "timestamp": timestamp,
            "query": f"{query} [Mock Run #{i+1}]",
            "domain_description": domain,
            "retrieved_contexts_count": random.randint(1, 4),
            "faithfulness_score": round(faithfulness, 4),
            "answer_relevancy_score": round(relevancy, 4),
            "context_precision_score": round(random.uniform(0.7, 1.0), 4),
            "deepeval_hallucination_score": round(faithfulness + random.uniform(-0.1, 0.1), 4),
            "deepeval_answer_relevancy_score": round(relevancy + random.uniform(-0.05, 0.05), 4),
            "total_cost_usd": round(cost, 6),
            "latency_ms": round(latency, 2),
            "route_decision": route_decision,
            "passed": passed,
            "failure_reasons": failure_reasons,
            "agent_reasoning": reason
        }
        
        # Clean scores bounds
        trace["deepeval_hallucination_score"] = max(0.0, min(1.0, trace["deepeval_hallucination_score"]))
        trace["deepeval_answer_relevancy_score"] = max(0.0, min(1.0, trace["deepeval_answer_relevancy_score"]))
        
        traces.append(trace)
        
    return traces


def seed_database():
    print("=" * 60)
    print("  SEEDING MOCK EVALUATION TRACES (500+ ENTRIES)  ")
    print("=" * 60)
    
    traces = generate_mock_traces(520) # Generate 520 records to be safe
    
    # 1. Try seeding into MongoDB
    db = get_mongo_db()
    if db is not None:
        try:
            print("[INFO] Connected to MongoDB Atlas. Seeding database...")
            # Clear existing collections to keep demo data clean
            db.traces.delete_many({})
            # Insert entries
            result = db.traces.insert_many(traces)
            print(f"[SUCCESS] Successfully seeded {len(result.inserted_ids)} records in MongoDB.")
            print("Database name      :", db.name)
            print("Collection name    : traces")
            print("=" * 60)
        except Exception as e:
            print(f"[WARNING] MongoDB write failed: {str(e)}. Falling back to JSONL file...")
            
    # 2. Local Fallback File seeding
    print("[INFO] Seeding into local fallback traces log...")
    log_dir = os.environ.get("EVAL_HARNESS_LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "traces.jsonl")
    
    try:
        # Convert any MongoDB ObjectId to string to prevent serialization errors
        for t in traces:
            if "_id" in t:
                t["_id"] = str(t["_id"])
        with open(log_path, "w", encoding="utf-8") as f:
            for t in traces:
                f.write(json.dumps(t) + "\n")
        print(f"[SUCCESS] Successfully seeded {len(traces)} records in local log file: {log_path}")
        print("=" * 60)
    except Exception as e:
        print(f"[ERROR] Seeding failed: {str(e)}")


if __name__ == "__main__":
    seed_database()
