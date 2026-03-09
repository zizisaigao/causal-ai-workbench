"""Application entrypoint for causal-ai-workbench MVP."""

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="causal-ai-workbench", version="0.1.0")
app.include_router(router, prefix="/api")
