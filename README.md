# agy-bridge

OpenAI-compatible HTTP bridge (FastAPI) that forwards chat completion requests to the Antigravity CLI (`agy`), so it can be used as a custom provider in Hermes.

- Listens on `127.0.0.1:8765`
- Default model `claude-sonnet-5-5-medium`, automatic fallback `gemini-3.8-flash-high`
- Supports streaming and non-streaming responses

## Setup
```
python3 -m venv ~/.hermes/agy-bridge-venv
~/.hermes/agy-bridge-venv/bin/pip install -r requirements.txt
python3 agy_bridge.py
```

## systemd user service
```
mkdir -p ~/.config/systemd/user
cp systemd/agy-bridge.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now agy-bridge.service
```
The unit targets `default.target` only and does not depend on `graphical-session.target`.

## Hermes provider
Point a custom provider at `http://127.0.0.1:8765/v1`.
