"""Entry point: start the FastAPI server that hosts the agent."""

import uvicorn

from app.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    print("\n" + "=" * 60)
    print("  AI Agent server starting...")
    print("  Open in your browser:  http://localhost:8000/docs")
    print("  Open WebUI URL to use: http://localhost:8000/v1")
    print("  (Do NOT type 0.0.0.0 in the browser — use localhost)")
    print("=" * 60 + "\n")
    uvicorn.run("app.api:app", host=settings.host, port=settings.port, reload=False)
