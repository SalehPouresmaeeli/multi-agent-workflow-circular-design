### Shared setup for all tests. pytest runs this file automatically, before any test file is imported.
###
### Several agents load db_config.json and create their LLM as soon as they are imported. On a machine
### without .env or db_config.json (e.g. GitHub's test servers) that import would fail, so every test run
### gets a fake API key and a temporary Neo4j config instead. This also guarantees that tests never use
### your real key or database. No LLM request is sent and no database connection is opened by importing.

import json
import os
import tempfile
from pathlib import Path

# Fake key: set before llm_config loads .env (load_dotenv does not overwrite variables that already exist)
os.environ["OPENROUTER_API_KEY"] = "fake-key-for-tests"

# Temporary db_config.json with placeholder connection details
_fake_db_config = Path(tempfile.mkdtemp()) / "db_config.json"
_fake_db_config.write_text(json.dumps({
    "uri": "neo4j://localhost:7687",
    "username": "neo4j",
    "password": "fake-password-for-tests",
    "database_name": "KnowledgeGraphTEST",
}))

from src.tools import db_config     # noqa: E402  (must come after the fake settings above)
db_config._CONFIG_PATH = _fake_db_config
