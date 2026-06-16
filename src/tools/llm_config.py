# To find correct path of files
import sys
from pathlib import Path
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
####################################

import json
import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

### Central LLM model config: To change which model an agent uses, edit llm_config.json
### in the project root — no code changes needed. To give an agent a different model than
### the shared default, add its key under "agents" in that file.
### Model names prefixed with "openrouter/" (e.g. "openrouter/google/gemini-3.8-flash") are
### routed through OpenRouter using OPENROUTER_API_KEY from .env; any other prefix
### (e.g. "gemini/...") goes straight to that provider as before.
_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "llm_config.json"
_FALLBACK_MODEL = "gemini/gemini-3.5-flash-lite"   # used only if llm_config.json is missing/unreadable
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_model(agent_key: str) -> str:
    """
    Looks up the model name to use for a given agent (e.g. "extraction_agent", "rag_agent")
    from llm_config.json. Resolution order:
      1. agents[agent_key]   — a model override specific to this agent
      2. default_model       — the shared fallback for any agent without its own override
      3. _FALLBACK_MODEL     — a hardcoded last resort if the config file can't be read at all
    """
    try:
        with open(_CONFIG_PATH, "r") as f:
            config = json.load(f)
        default_model = config.get("default_model", _FALLBACK_MODEL)
        return config.get("agents", {}).get(agent_key, default_model)
    except Exception as e:
        print(f"Warning: Could not read llm_config.json. Using default model. ({e})")
        return _FALLBACK_MODEL


def get_llm(agent_key: str, **kwargs) -> LLM:
    """
    Builds the CrewAI LLM for a given agent, using the model from get_model(agent_key).
    Extra kwargs (e.g. temperature=0) are passed straight through to LLM.
    For "openrouter/..." models, the OpenRouter API key and base URL are attached explicitly.
    """
    model = get_model(agent_key)

    if model.startswith("openrouter/"):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(f"'{agent_key}' is set to use {model}, but OPENROUTER_API_KEY is not set in .env")
        return LLM(model=model, api_key=api_key, base_url=_OPENROUTER_BASE_URL, **kwargs)

    return LLM(model=model, **kwargs)


### Quick connectivity check: run "python .\src\tools\llm_config.py" from the root folder
if __name__ == "__main__":
    agent_key = sys.argv[1] if len(sys.argv) > 1 else "rag_agent"
    llm = get_llm(agent_key, temperature=0)
    print(f"Agent: {agent_key} | Model: {llm.model}")
    print("Response:", llm.call("Reply with exactly: OK"))
