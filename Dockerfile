FROM python:3.11-slim

WORKDIR /app

# Core deps first for more stable builds on HF Spaces.
RUN pip install --no-cache-dir "openenv-core>=0.1.0" "fastapi>=0.104.0" "uvicorn>=0.24.0"

# This Dockerfile assumes the HF Space repo root is this env directory.
# Copy the full package into a package-named folder for clean imports.
COPY . ./reward_hack_honeypot_env/

RUN pip install --no-cache-dir ./reward_hack_honeypot_env

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r=httpx.get('http://localhost:8000/health', timeout=3); exit(0 if r.status_code==200 else 1)" || exit 1

CMD ["uvicorn", "reward_hack_honeypot_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
