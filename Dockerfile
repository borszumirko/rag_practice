FROM python:3.13-slim

WORKDIR /src/app

# Create non-root user early so we can own /app
RUN useradd --create-home appuser && chown appuser:appuser /src/app

# Install uv (fast Python package manager)
RUN pip install uv

# Copy dependency files first (Docker layer caching)
COPY --chown=appuser:appuser pyproject.toml .
COPY --chown=appuser:appuser uv.lock* .

# Switch to non-root user before installing deps (so .venv is owned by appuser)
USER appuser

# Install dependencies without building the project before its source is copied
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY --chown=appuser:appuser src/app/ src/app/
COPY --chown=appuser:appuser README.md .

# Install the project now that its source and metadata are available
RUN uv sync --frozen --no-dev

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run uvicorn directly from venv (avoids uv re-syncing at runtime)
CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
