"""
API Request and Response Models
Pydantic models for input validation and response structure.
"""

from pydantic import BaseModel, Field
from datetime import datetime, timezone


class ChatRequest(BaseModel):
    """Incoming chat request."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The question to answer from the indexed Wikipedia corpus",
    )
    thread_id: str = Field(default="default", description="Client correlation ID; requests are independent questions")


class Source(BaseModel):
    """A retained evidence slice, with end-exclusive source character offsets."""

    chunk_id: str
    document_id: str
    title: str
    url: str
    start_char: int
    end_char: int


class ChatResponse(BaseModel):
    """Chat response returned to the client."""

    response: str
    thread_id: str
    model_used: str
    cached: bool = False
    processing_time_ms: float
    security_notes: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    retrieval_ms: float = 0
    input_tokens: int = 0
    output_tokens: int = 0
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    environment: str
    version: str = "1.1.1"
    checks: dict[str, bool] = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    """Metrics endpoint response."""

    total_requests: int
    total_errors: int
    error_rate: str
    avg_latency_ms: float
    cache_hit_rate: str
    total_input_tokens: int
    total_output_tokens: int


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None
    request_id: str | None = None
