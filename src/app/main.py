"""FastAPI serving the experiment RAG pipeline, configured and launched by Hydra."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from contextvars import copy_context
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langsmith import tracing_context
from omegaconf import DictConfig, OmegaConf
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
import uvicorn

from app.cache import ResponseCache
from app.config import load_config
from app.models import ChatRequest, ChatResponse, HealthResponse, MetricsResponse, Source
from app.monitoring import MetricsCollector, RequestTimer, get_logger
from app.pipeline import RAGPipeline
from app.security import SecurityPipeline


logger = get_logger()


def create_app(cfg: DictConfig | None = None, output: Path | None = None) -> FastAPI:
    """Build an isolated app; load models and the existing index during startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = cfg if cfg is not None else load_config()
        app.state.cfg = settings
        logger.setLevel(str(settings.server.log_level).upper())
        folder = output or Path(to_absolute_path("data/runs/server")) / datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%d_%H-%M-%S-%f")
        folder.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(settings, folder / "resolved_config.yaml", resolve=True)
        app.state.security = SecurityPipeline()
        app.state.cache = ResponseCache(ttl_seconds=settings.server.cache_ttl_seconds)
        app.state.metrics = MetricsCollector()
        # SQLite connections belong to the thread that creates them. Serialize all
        # RAG work on one worker, leaving the ASGI event loop free for health checks.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag")
        app.state.executor = executor
        app.state.pipeline = None
        try:
            app.state.pipeline = await asyncio.get_running_loop().run_in_executor(
                executor, partial(RAGPipeline, settings, folder)
            )
            logger.info("RAG API ready", extra={"extra_data": {
                "generation_model": settings.generation.model,
                "embedding_model": settings.embedding.model,
                "retrieval": settings.retrieval.method,
            }})
            yield
        finally:
            try:
                if app.state.pipeline is not None:
                    await asyncio.get_running_loop().run_in_executor(
                        executor, app.state.pipeline.close
                    )
            finally:
                app.state.pipeline = None
                executor.shutdown(wait=True)
            logger.info("RAG API stopped", extra={"extra_data": app.state.metrics.summary})

    application = FastAPI(
        title="Natural Questions RAG API",
        description="Question-only retrieval and evidence-based answer generation.",
        version="1.0.0",
        lifespan=lifespan,
    )
    # Resolve the rate limit at request time, after Hydra configuration is loaded.
    limiter = Limiter(key_func=get_remote_address)
    application.state.limiter = limiter

    @application.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(status_code=429, content={
            "error": "Rate limit exceeded",
            "detail": "Too many requests. Please slow down.",
        })

    def rate_limit():
        return application.state.cfg.server.rate_limit

    @application.post("/chat", response_model=ChatResponse)
    @limiter.limit(rate_limit)
    async def chat(request: Request, body: ChatRequest):
        state = request.app.state
        with tracing_context(enabled=state.cfg.server.tracing_enabled):
            with RequestTimer() as timer:
                allowed, question, notes = state.security.check_input(body.message)
                if not allowed or not question.strip():
                    state.metrics.record_request(latency_ms=0, error=True)
                    raise HTTPException(status_code=400, detail="Question blocked or empty after input validation.")

                saved = state.cache.get(question)
                if saved is not None:
                    state.metrics.record_request(latency_ms=0, cache_hit=True)
                    response = ChatResponse.model_validate_json(saved)
                    return response.model_copy(update={
                        "thread_id": body.thread_id, "cached": True,
                        "processing_time_ms": 0,
                        "security_notes": notes + response.security_notes,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

                try:
                    result = await asyncio.get_running_loop().run_in_executor(
                        state.executor, copy_context().run,
                        partial(state.pipeline.invoke, question),
                    )
                except Exception:
                    logger.exception("RAG request failed")
                    state.metrics.record_request(latency_ms=0, error=True)
                    raise HTTPException(status_code=500, detail="An error occurred while processing your request.") from None

                text, warnings = state.security.check_output(result["answer"])
                usage = result.get("usage", {})
                sources = [Source(chunk_id=c["chunk_id"], **{
                    key: c["metadata"][key] for key in
                    ("document_id", "title", "url", "start_char", "end_char")
                }) for c in result["chunks"]]
            response = ChatResponse(
                response=text, thread_id=body.thread_id,
                model_used=result.get("model") or state.cfg.generation.model,
                processing_time_ms=round(timer.elapsed_ms, 2),
                security_notes=notes + warnings, sources=sources,
                retrieval_ms=result["retrieval_ms"],
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )
            # Cache output metadata; input security notes belong to this request.
            state.cache.set(question, response.model_copy(update={
                "security_notes": warnings,
            }).model_dump_json())
            state.metrics.record_request(
                latency_ms=timer.elapsed_ms, input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
            return response

    @application.get("/health", response_model=HealthResponse)
    async def health():
        state = application.state
        checks = {"rag": state.pipeline is not None, "security": state.security is not None,
                  "cache": state.cache is not None}
        return HealthResponse(status="healthy" if all(checks.values()) else "degraded",
                              environment=state.cfg.server.environment, checks=checks)

    @application.get("/metrics", response_model=MetricsResponse)
    async def get_metrics():
        return MetricsResponse(**application.state.metrics.summary)

    @application.get("/cache/stats")
    async def cache_stats():
        return application.state.cache.stats

    return application


# ASGI imports also compose server.yaml through Hydra during lifespan startup.
app = create_app()


@hydra.main(version_base="1.3", config_path="conf", config_name="server")
def main(cfg: DictConfig):
    """Serve by default, or run chunks/index/ask/evaluate with the same settings."""
    if cfg.command != "serve":
        from app.rag import run
        run(cfg)
        return
    output = Path(HydraConfig.get().runtime.output_dir)
    uvicorn.run(create_app(cfg, output), host=cfg.server.host,
                port=int(cfg.server.port), log_level=cfg.server.log_level)


if __name__ == "__main__":
    main()
