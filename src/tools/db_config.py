# To find correct path of files
import sys
from pathlib import Path
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
####################################

import json

### Central Neo4j connection config: credentials live in db_config.json in the project root
### (git-ignored). Copy db_config.example.json to db_config.json and fill in your own values.
### The password has no default on purpose — a missing password should fail loudly here,
### not as a confusing authentication error from Neo4j later on.
_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "db_config.json"


def load_db_config() -> dict:
    """
    Reads Neo4j connection details from db_config.json and returns a dict with keys
    uri, username, password, database_name. Non-secret fields fall back to local defaults;
    the password is required.
    """
    try:
        with open(_CONFIG_PATH, "r") as f:
            db_config = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(
            f"db_config.json not found at {_CONFIG_PATH}. "
            "Copy db_config.example.json to db_config.json and fill in your Neo4j password."
        )

    password = db_config.get("password")
    if not password:
        raise RuntimeError("db_config.json has no 'password' set. Add your Neo4j password to it.")

    return {
        "uri": db_config.get("uri", "neo4j://127.0.0.1:7687"),
        "username": db_config.get("username", "neo4j"),
        "password": password,
        "database_name": db_config.get("database_name", "KnowledgeGraph"),
    }
