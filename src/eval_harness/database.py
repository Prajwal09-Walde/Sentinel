import os
from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database

_client: Optional[MongoClient] = None


def get_mongo_db() -> Optional[Database]:
    """Retrieve or initialize the MongoDB database instance if MONGO_URI is set.
    
    Returns:
        pymongo.database.Database instance if successful, else None.
    """
    global _client
    mongo_uri = os.environ.get("MONGO_URI")
    db_name = os.environ.get("MONGO_DB_NAME", "eval_harness")

    if not mongo_uri:
        return None

    if _client is None:
        try:
            # Configure with a 2-second timeout for fast failover/offline checks
            _client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
        except Exception:
            return None

    try:
        # Verify connection by triggering a simple admin command
        _client.admin.command('ping')
        return _client[db_name]
    except Exception:
        # If connection fails, return None to trigger fallback logging
        return None
