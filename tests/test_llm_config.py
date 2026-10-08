### Tests for src/tools/llm_config.py — choosing each agent's model and building its LLM.
### A temporary llm_config.json and fake API keys are used, so no real key is needed and no LLM is called.

import json
import pytest
from src.tools import llm_config


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    """Returns a function that writes a temporary llm_config.json and points the loader at it."""
    path = tmp_path / "llm_config.json"
    monkeypatch.setattr(llm_config, "_CONFIG_PATH", path)

    def write(contents: dict):
        path.write_text(json.dumps(contents))
    return write


# --- get_model: which model name each agent gets ---

def test_agent_with_its_own_entry_gets_that_model(config_file):
    config_file({"default_model": "gemini/default-model", "agents": {"rag_agent": "openrouter/google/special-model"}})

    assert llm_config.get_model("rag_agent") == "openrouter/google/special-model"


def test_agent_without_entry_gets_default_model(config_file):
    config_file({"default_model": "gemini/default-model", "agents": {"rag_agent": "openrouter/google/special-model"}})

    assert llm_config.get_model("mentor_agent") == "gemini/default-model"


def test_config_without_default_model_uses_fallback(config_file):
    config_file({"agents": {}})

    assert llm_config.get_model("mentor_agent") == llm_config._FALLBACK_MODEL


def test_missing_config_file_uses_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_config, "_CONFIG_PATH", tmp_path / "does_not_exist.json")

    assert llm_config.get_model("rag_agent") == llm_config._FALLBACK_MODEL


# --- get_llm: building the LLM object (no request is sent) ---

def test_openrouter_model_without_key_gives_clear_error(config_file, monkeypatch):
    config_file({"default_model": "openrouter/google/gemini-3.8-flash"})
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        llm_config.get_llm("rag_agent")


def test_openrouter_model_uses_openrouter_key_and_address(config_file, monkeypatch):
    config_file({"default_model": "openrouter/google/gemini-3.8-flash"})
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-openrouter-key")

    llm = llm_config.get_llm("rag_agent", temperature=0)

    assert llm.api_key == "fake-openrouter-key"
    assert llm.base_url == "https://openrouter.ai/api/v1"
    assert llm.temperature == 0
