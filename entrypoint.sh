#!/bin/bash
# Start Ollama in the background
ollama serve &

echo "[Bifrost] Waiting for Ollama daemon to become ready..."
max_wait=60
started_at=$(date +%s)
while true; do
  if curl -s http://127.0.0.1:11434/api/tags > /dev/null; then
    ended_at=$(date +%s)
    echo "[Bifrost] Ollama is ready. Startup took $((ended_at - started_at)) seconds."
    break
  fi
  now=$(date +%s)
  if [ $((now - started_at)) -gt $max_wait ]; then
    echo "[Bifrost] ERROR: Ollama daemon failed to become ready within ${max_wait}s!"
    break
  fi
  sleep 0.2
done

# Run the Bifrost main application, passing along any arguments (like --serve)
exec python -m app.main "$@"
