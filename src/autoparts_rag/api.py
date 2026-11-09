"""HTTP API: POST /chat (SSE), POST /feedback, GET /pdf/{id}, GET /health; UI: / і /evals."""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from .db import connection
from .pipeline import ChatRequest, Pipeline
from .web import router as web_router

app = FastAPI(title="Autoparts RAG support")
app.include_router(web_router)


class ChatIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=2000)
    product_article: str | None = Field(default=None, max_length=64)
    product_manufacturer: str | None = Field(default=None, max_length=128)


class FeedbackIn(BaseModel):
    trace_id: int
    useful: bool
    comment: str | None = Field(default=None, max_length=2000)


@app.on_event("startup")
def warm_up() -> None:
    # перше завантаження bge-m3 і reranker займає секунди, тож робимо його до першого запиту
    from .models.embedder import get_embedder
    from .models.reranker import get_reranker

    get_embedder()
    get_reranker()


@app.post("/chat")
async def chat(body: ChatIn) -> EventSourceResponse:
    req = ChatRequest(body.session_id, body.message, body.product_article, body.product_manufacturer)

    async def events():
        async for ev in Pipeline().run(req):
            yield {"event": ev["event"], "data": json.dumps(ev["data"], ensure_ascii=False)}

    return EventSourceResponse(events())


@app.post("/feedback")
def feedback(body: FeedbackIn) -> dict:
    with connection() as conn:
        exists = conn.execute("SELECT 1 FROM traces WHERE id = %s", (body.trace_id,)).fetchone()
        if not exists:
            raise HTTPException(404, "trace not found")
        conn.execute("INSERT INTO feedback (trace_id, useful, comment) VALUES (%s, %s, %s)",
                     (body.trace_id, body.useful, body.comment))
    return {"ok": True}


@app.get("/pdf/{document_id}")
def pdf(document_id: int) -> FileResponse:
    with connection() as conn:
        row = conn.execute("SELECT storage_path, file_name FROM documents WHERE id = %s",
                           (document_id,)).fetchone()
    if not row or not Path(row["storage_path"]).exists():
        raise HTTPException(404, "document not found")
    return FileResponse(row["storage_path"], media_type="application/pdf",
                        headers={"Content-Disposition": f'inline; filename="{row["file_name"]}"'})


@app.get("/health")
def health() -> dict:
    with connection() as conn:
        docs = conn.execute("SELECT count(*) AS n FROM documents WHERE is_active").fetchone()["n"]
    return {"status": "ok", "active_documents": docs}
