### Tests for src/tools/db_config.py — loading the Neo4j connection details.
### Each test points the loader at its own temporary db_config.json (tmp_path), so your real file is never read.

import json
import pytest
from src.tools import db_config


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    """Returns a function that writes a temporary db_config.json and points the loader at it."""
    path = tmp_path / "db_config.json"
    monkeypatch.setattr(db_config, "_CONFIG_PATH", path)

    def write(contents: dict):
        path.write_text(json.dumps(contents))
    return write


def test_reads_all_fields(config_file):
    config_file({
        "uri": "neo4j://example:7687",
        "username": "alice",
        "password": "secret",
        "database_name": "TestGraph",
    })

    assert db_config.load_db_config() == {
        "uri": "neo4j://example:7687",
        "username": "alice",
        "password": "secret",
        "database_name": "TestGraph",
    }


def test_non_secret_fields_fall_back_to_defaults(config_file):
    config_file({"password": "secret"})

    result = db_config.load_db_config()

    assert result["uri"] == "neo4j://127.0.0.1:7687"
    assert result["username"] == "neo4j"
    assert result["database_name"] == "KnowledgeGraph"


def test_missing_file_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(db_config, "_CONFIG_PATH", tmp_path / "does_not_exist.json")

    with pytest.raises(RuntimeError, match="db_config.example.json"):
        db_config.load_db_config()


@pytest.mark.parametrize("contents", [{"username": "neo4j"}, {"password": ""}])
def test_missing_or_empty_password_gives_clear_error(config_file, contents):
    config_file(contents)

    with pytest.raises(RuntimeError, match="password"):
        db_config.load_db_config()
