Model-3 supports HTTP-first (OLLAMA_URL) and CLI fallback (local ollama binary).

Environment variables:
- OLLAMA_URL: URL to Ollama HTTP API (http://host.docker.internal:11434)
- MODEL3_MODEL_NAME: model name (e.g., mistral, phi)
- MODEL3_TIMEOUT_SECONDS: HTTP timeout in seconds
- MODEL3_MAX_TOKENS: max tokens for HTTP call
- MODEL3_CLI_FALLBACK: "1" or "true" to enable local ollama CLI fallback

Local (Windows) recommended workflow:
- enable CLI fallback: set MODEL3_CLI_FALLBACK=1
- ensure `ollama` is in PATH (installed locally)
- run `python model3_runner.py --input /path/to/model2.json --out_dir /path/to/out`
