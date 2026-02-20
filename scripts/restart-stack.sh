#!/bin/bash
# Stop, rebuild with no cache, and restart the full stack.
# Run from project root: ./scripts/restart-stack.sh
#
# Options:
#   -p, --prune   Free Docker disk space before building (run if you hit "no space left on device")
#
# Data safety: Prune and down do NOT remove volumes. Agents, users, and DB
# stay in postgres_data; uploads in uploads_data. Only images/cache are pruned.

set -e
cd "$(dirname "$0")/.."

PRUNE=0
for arg in "$@"; do
  case "$arg" in
    -p|--prune) PRUNE=1 ;;
  esac
done

echo "Stopping all services..."
docker compose down

echo "Removing dangling images (<none>)..."
docker image prune -f

if [ "$PRUNE" = 1 ]; then
  echo "Freeing Docker disk space (build cache, unused images; volumes are NOT touched)..."
  docker builder prune -a -f
  docker system prune -f
fi

echo "Rebuilding frontend, backend, worker, and chat-ui with --no-cache..."
docker compose build --no-cache frontend backend worker chat-ui

echo "Pulling latest LiteLLM image..."
docker compose pull litellm

echo "Starting stack..."
docker compose up -d

echo ""
echo "Waiting for LiteLLM to be ready..."
for i in {1..30}; do
  if curl -s -o /dev/null -w "%{http_code}" http://localhost:4000/health 2>/dev/null | grep -q 200; then
    echo "LiteLLM is up."
    break
  fi
  sleep 2
  [ $i -eq 30 ] && echo "LiteLLM health check timed out (may still be starting)."
done

echo ""
echo "Waiting for backend to be ready..."
for i in {1..30}; do
  if curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/health 2>/dev/null | grep -q 200; then
    echo "Backend is up."
    break
  fi
  sleep 2
  [ $i -eq 30 ] && echo "Backend health check timed out (may still be starting)."
done

echo ""
echo "Stack is running:"
echo "  - Admin UI (data domains, monitoring): http://localhost:5174/"
echo "  - User-facing Chat UI (chat, translation): http://localhost:7860/"
echo "  - Backend API:                           http://localhost:3000"
echo "  - LiteLLM proxy:                         http://localhost:4000"
echo ""
echo "Do a hard refresh in the browser (Cmd+Shift+R or Ctrl+Shift+R) to avoid cached assets."
