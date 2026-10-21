"""FastAPI application exposing the agent, OpenAI-compatible for Open WebUI."""

import logging
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agent import OllamaConnectionError, chat
from app.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Agent Task Automation System", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    conversation_id: str = "default"


class ChatResponse(BaseModel):
    response: str


class OAIMessage(BaseModel):
    role: str
    content: str


class OAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list[OAIMessage]
    conversation_id: Optional[str] = Field(default=None, alias="conversation_id")
    user: Optional[str] = None
    stream: bool = False

    model_config = {"populate_by_name": True, "extra": "ignore"}


def _handle_chat(message: str, conversation_id: str) -> str:
    try:
        return chat(message, conversation_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OllamaConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error while processing chat")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


@app.get("/", include_in_schema=False)
def root():
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/docs")


@app.get("/v1", include_in_schema=False)
def v1_root():
    return {
        "message": "OpenAI-compatible API. Use POST /v1/chat/completions and GET /v1/models.",
        "model": get_settings().ollama_model,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    logger.info("POST /chat message=%r conversation_id=%r", request.message, request.conversation_id)
    return ChatResponse(response=_handle_chat(request.message, request.conversation_id))


@app.get("/v1/models")
def list_models():
    settings = get_settings()
    return {
        "object": "list",
        "data": [
            {
                "id": settings.ollama_model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local-ollama",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: OAIChatRequest):
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided.")
    message = user_messages[-1].content
    conversation_id = request.conversation_id or request.user or "default"
    logger.info("POST /v1/chat/completions conversation_id=%r", conversation_id)
    try:
        answer = _handle_chat(message, conversation_id)
    except HTTPException as exc:
        # Open WebUI expects OpenAI-style errors; return JSON with controlled status
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": exc.detail, "type": "api_error"}},
        )

    settings = get_settings()
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model or settings.ollama_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
