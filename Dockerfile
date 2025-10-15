FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install uv for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create app directory and user
RUN useradd -m -d /app hyrl-user
WORKDIR /app

# Copy all necessary files for build
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY Makefile ./
COPY protos/ ./protos/

# Install dependencies (uv will use CPU-only PyTorch from tool.uv.sources)
RUN uv sync && \
    # Clean up cache to reduce image size
    uv cache clean

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