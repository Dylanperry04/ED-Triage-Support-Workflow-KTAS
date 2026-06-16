#!/usr/bin/env bash
set -e

export PYTHONUNBUFFERED=1
export PORT="${PORT:-8000}"

echo "Starting KTAS research API on port ${PORT}"
exec gunicorn -w 2 -k uvicorn.workers.UvicornWorker --bind=0.0.0.0:${PORT} --timeout 120 app.main:app
