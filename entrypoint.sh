#!/bin/bash
# Start Ollama in the background
ollama serve &

# Wait a few seconds for the daemon to spin up
sleep 3

# Run the Bifrost main application, passing along any arguments (like --serve)
exec python -m app.main "$@"
