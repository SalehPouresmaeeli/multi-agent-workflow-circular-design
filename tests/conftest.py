### Shared setup for all tests. pytest runs this file automatically, before any test file is imported.
###
### Several agents load db_config.json and create their LLM as soon as they are imported. On a machine
### without .env or db_config.json (e.g. GitHub's test servers) that import would fail, so a normal test
### run gets a fake API key and a temporary Neo4j config instead. This also guarantees that normal tests
### never use your real key or database.
###
### Integration runs (pytest -m integration) are different: they use your real db_config.json, and every
### agent uses one Gemini model through Google AI Studio (GEMINI_API_KEY in .env) instead of the models in
### llm_config.json. Set INTEGRATION_TEST_MODEL to use a different model for those runs.

import json
import os
import tempfile
from pathlib import Path

DEFAULT_INTEGRATION_MODEL = "gemini/gemini-3.5-flash-lite"


def is_integration_run(config) -> bool:
    """True when pytest was started with -m integration (the default is -m "not integration")."""
    mark_expression = config.option.markexpr or ""
    return "integration" in mark_expression and "not integration" not in mark_expression


def pytest_configure(config):
    # Runs before any test file is imported, so these settings are in place before the agents load.
    from src.tools import db_config, llm_config

    if is_integration_run(config):
        # Temporary llm_config.json: every agent uses the integration model (your real file is not changed)
        model = os.getenv("INTEGRATION_TEST_MODEL", DEFAULT_INTEGRATION_MODEL)
        integration_llm_config = Path(tempfile.mkdtemp()) / "llm_config.json"
        integration_llm_config.write_text(json.dumps({"default_model": model, "agents": {}}))
        llm_config._CONFIG_PATH = integration_llm_config
        return

    # Fake key, so agents that use OpenRouter can be created without a real one
    os.environ["OPENROUTER_API_KEY"] = "fake-key-for-tests"

    # Temporary db_config.json with placeholder connection details
    fake_db_config = Path(tempfile.mkdtemp()) / "db_config.json"
    fake_db_config.write_text(json.dumps({
        "uri": "neo4j://localhost:7687",
        "username": "neo4j",
        "password": "fake-password-for-tests",
        "database_name": "KnowledgeGraphTEST",
    }))
    db_config._CONFIG_PATH = fake_db_config
