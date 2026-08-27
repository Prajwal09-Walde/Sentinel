import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

mongo_uri = os.environ.get("MONGO_URI")
db_name = os.environ.get("MONGO_DB_NAME", "eval_harness")

print("=" * 60)
print("TESTING BYPASS SSL HANDSHAKE")
print("=" * 60)

try:
    print("[INFO] Attempting to connect with tlsAllowInvalidCertificates=True...")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000, tlsAllowInvalidCertificates=True)
    client.admin.command("ping")
    print("[SUCCESS] Connected successfully!")
    db = client[db_name]
    count = db.traces.count_documents({})
    print(f"[INFO] Documents count: {count}")
except Exception as e:
    print(f"[ERROR] Failed: {str(e)}")
print("=" * 60)
