from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import asyncio
import json
import pandas as pd
import io
import uvicorn

### web app for knowledge graph: To run it, use "uvicorn web.web_app_KG:app --reload" on the root. 
### Then open web browser and go to http://localhost:8000.
#################################################

from src.agents.ingestion_pipeline import run_ingestion_pipeline   # Import bridge function
from src.agents.rag_agent import run_rag_pipeline               

app = FastAPI(title="Knowledge Graph API")

# Dynamically locate the directory this app.py file lives in (the web/ folder)
BASE_DIR = Path(__file__).resolve().parent

# This tells the server: "Any web request starting with /static should look inside the local static/ folder"
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

class QueryRequest(BaseModel):
    user_question: str

@app.get("/")
async def serve_ui():
    """Serves the frontend UI."""
    # Build the absolute path to index.html to prevent file-not-found errors
    html_path = BASE_DIR / "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

def _sse_stream(q: asyncio.Queue):
    """Async generator that reads events from a queue and yields SSE-formatted strings."""
    async def generate():
        while True:
            event = await q.get()
            if event is None:
                return
            yield f"data: {json.dumps(event)}\n\n"
    return generate()

def _sse_response(generator):
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    sheet_name: str = Form(default=None)
):
    """
    Parses the uploaded file, then streams real-time SSE events from the
    CrewAI ingestion pipeline (extraction -> canonicalisation -> ingestion).
    """
    # --- File parsing happens before the stream opens so errors return clean HTTP 4xx ---
    contents = await file.read()
    filename = file.filename.lower()

    if filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(contents))
    elif filename.endswith(('.xlsx', '.xlsm')):
        target_sheet = sheet_name if sheet_name else 0
        try:
            df = pd.read_excel(io.BytesIO(contents), sheet_name=target_sheet)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Sheet '{sheet_name}' not found in the uploaded Excel file.")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or Excel file.")

    df = df.replace([float('inf'), float('-inf')], "").fillna("")
    column_headers = df.columns.tolist()
    excel_records = df.to_dict(orient="records")

    # --- Stream pipeline progress as SSE ---
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def push(event):
        loop.call_soon_threadsafe(q.put_nowait, event)

    async def run():
        try:
            await loop.run_in_executor(
                None,
                lambda: run_ingestion_pipeline(excel_records, column_headers, event_callback=push)
            )
        except Exception as e:
            push({"type": "error", "message": f"Pipeline error: {str(e)}"})
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)

    asyncio.create_task(run())
    return _sse_response(_sse_stream(q))


@app.post("/api/query")
async def query_graph(request: QueryRequest):
    """
    Streams real-time SSE events from the CrewAI RAG agent as it writes
    and executes Cypher queries, then emits the final answer as a result event.
    """
    if not request.user_question.strip():
        raise HTTPException(status_code=400, detail="The question cannot be empty.")

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def push(event):
        loop.call_soon_threadsafe(q.put_nowait, event)

    async def run():
        try:
            await loop.run_in_executor(
                None,
                lambda: run_rag_pipeline(request.user_question, event_callback=push)
            )
        except Exception as e:
            push({"type": "error", "message": f"Query error: {str(e)}"})
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)

    asyncio.create_task(run())
    return _sse_response(_sse_stream(q))

if __name__ == "__main__":
    # Note: If running this script directly, uvicorn runs here
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)