### Tests for web/web_app_KG.py — the knowledge graph web app's pages and API.
### FastAPI's TestClient sends requests to the app in memory (no server is started). The ingestion and
### RAG pipelines are replaced by stand-ins, so no LLM, agent or database is used.

import io
import json
import pandas as pd
import pytest
from fastapi.testclient import TestClient
import web.web_app_KG as web_app


@pytest.fixture
def client():
    return TestClient(web_app.app)


def sse_events(response):
    """Turns a server-sent-events response body into a list of event dicts."""
    return [json.loads(line[len("data: "):]) for line in response.text.splitlines() if line.startswith("data: ")]


def excel_bytes(sheets: dict) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, rows in sheets.items():
            pd.DataFrame(rows).to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()


# --- Main page ---

def test_main_page_is_served(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "Knowledge Graph Agentic Tool" in response.text


# --- Upload: file checks (these happen before any pipeline runs) ---

def test_unsupported_file_type_is_rejected(client):
    response = client.post("/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")})

    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_unknown_excel_sheet_is_rejected(client):
    content = excel_bytes({"Guidelines": [{"DfReX": "Repair"}]})

    response = client.post(
        "/api/upload",
        files={"file": ("guidelines.xlsx", content)},
        data={"sheet_name": "NoSuchSheet"},
    )

    assert response.status_code == 400
    assert "NoSuchSheet" in response.json()["detail"]


# --- Upload: hand-off to the ingestion pipeline ---

@pytest.fixture
def fake_ingestion(monkeypatch):
    """Replaces the ingestion pipeline and records what the web app passed to it."""
    calls = []

    def fake_run_ingestion_pipeline(excel_records, column_headers, event_callback=None):
        calls.append({"records": excel_records, "headers": column_headers})
        event_callback({"type": "done", "message": "Pipeline complete."})
        return "ok"

    monkeypatch.setattr(web_app, "run_ingestion_pipeline", fake_run_ingestion_pipeline)
    return calls


def test_csv_rows_are_passed_to_the_pipeline(client, fake_ingestion):
    csv = b"DfReX,Ref,Relevance\nRemanufacturing,[4],High\nRepair,,Medium\n"

    response = client.post("/api/upload", files={"file": ("guidelines.csv", csv, "text/csv")})

    assert response.status_code == 200
    assert fake_ingestion[0]["headers"] == ["DfReX", "Ref", "Relevance"]
    assert fake_ingestion[0]["records"] == [
        {"DfReX": "Remanufacturing", "Ref": "[4]", "Relevance": "High"},
        {"DfReX": "Repair", "Ref": "", "Relevance": "Medium"},      # empty cells become "" rather than NaN
    ]
    assert sse_events(response) == [{"type": "done", "message": "Pipeline complete."}]


def test_chosen_excel_sheet_is_used(client, fake_ingestion):
    content = excel_bytes({
        "Notes": [{"Comment": "not this one"}],
        "Guidelines": [{"DfReX": "Recycling", "Relevance": "Low"}],
    })

    client.post("/api/upload", files={"file": ("guidelines.xlsx", content)}, data={"sheet_name": "Guidelines"})

    assert fake_ingestion[0]["records"] == [{"DfReX": "Recycling", "Relevance": "Low"}]


def test_pipeline_failure_is_reported_as_error_event(client, monkeypatch):
    def failing_pipeline(excel_records, column_headers, event_callback=None):
        raise RuntimeError("Ingestion failed, so the graph was not updated.")
    monkeypatch.setattr(web_app, "run_ingestion_pipeline", failing_pipeline)

    response = client.post("/api/upload", files={"file": ("guidelines.csv", b"DfReX\nRepair\n", "text/csv")})

    events = sse_events(response)
    assert events[-1]["type"] == "error"
    assert "graph was not updated" in events[-1]["message"]


# --- Query ---

@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_is_rejected(client, question):
    response = client.post("/api/query", json={"user_question": question})

    assert response.status_code == 400


def test_question_is_passed_to_rag_agent_and_answer_streamed(client, monkeypatch):
    asked = []

    def fake_run_rag_pipeline(user_question, event_callback=None):
        asked.append(user_question)
        event_callback({"type": "result", "answer": "Use screws — [4] (Academic paper, High relevance)"})
        return "done"
    monkeypatch.setattr(web_app, "run_rag_pipeline", fake_run_rag_pipeline)

    response = client.post("/api/query", json={"user_question": "How should the casing be joined?"})

    assert asked == ["How should the casing be joined?"]
    assert sse_events(response) == [{"type": "result", "answer": "Use screws — [4] (Academic paper, High relevance)"}]
