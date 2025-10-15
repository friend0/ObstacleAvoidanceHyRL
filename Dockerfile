FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create app directory and user
RUN useradd -m -d /app hyrl-user
WORKDIR /app

# Copy source code
COPY --chown=hyrl-user:hyrl-user src/ ./src/
COPY --chown=hyrl-user:hyrl-user pyproject.toml ./
COPY --chown=hyrl-user:hyrl-user Makefile ./
COPY --chown=hyrl-user:hyrl-user protos/ ./protos/

# Install Python dependencies as root (where uv is installed)
RUN uv sync

# Fix ownership of .venv
RUN chown -R hyrl-user:hyrl-user /app

# Switch to non-root user
USER hyrl-user

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PATH="/app/.venv/bin:$PATH"

# Expose the gRPC port
EXPOSE 50051

# Run the server
CMD ["python", "-m", "rl_policy.server"]