#!/usr/bin/env python3

import json
import subprocess
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Any
import uvicorn


app = FastAPI()

# Primary model and automatic fallback for the bridge service.
DEFAULT_MODEL = "claude-sonnet-5-5-medium"
FALLBACK_MODEL = "gemini-3.8-flash-high"


class Message(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    model: Optional[str] = DEFAULT_MODEL
    messages: List[Message]
    temperature: Optional[float] = None
    stream: Optional[bool] = False


def content_to_text(content):
    """
    Convert OpenAI-style message content into plain text.
    Handles both ordinary strings and multipart content.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))

        return "\n".join(parts)

    if content is None:
        return ""

    return str(content)


def build_prompt(messages):
    """
    Flatten the Hermes/OpenAI conversation into a prompt for agy.
    Prunes older message history from the middle if the prompt exceeds
    the CLI's STDIN truncation threshold (~145,000 chars), ensuring the
    system prompt and latest conversation turns are always preserved.
    """
    if not messages:
        return ""

    # Always preserve the system prompt (first message if role == 'system')
    has_system = messages[0].role.lower() == "system"
    system_part = f"===== SYSTEM =====\n{content_to_text(messages[0].content)}" if has_system else None
    rest_messages = messages[1:] if has_system else messages

    max_chars = 145000
    system_len = len(system_part) if system_part else 0
    available_chars = max(20000, max_chars - system_len)

    # Accumulate messages from newest to oldest
    selected_parts = []
    current_chars = 0
    truncated_history = False

    for message in reversed(rest_messages):
        role = message.role.upper()
        text = content_to_text(message.content)
        part = f"===== {role} =====\n{text}"
        part_len = len(part) + 2  # account for \n\n separator

        if current_chars + part_len > available_chars and selected_parts:
            truncated_history = True
            break

        selected_parts.append(part)
        current_chars += part_len

    selected_parts.reverse()

    parts = []
    if system_part:
        parts.append(system_part)
    if truncated_history:
        parts.append("===== SYSTEM =====\n[Note: Earlier conversation turns compacted to preserve context within CLI limits.]")
    parts.extend(selected_parts)

    return "\n\n".join(parts)


def call_agy_once(request, model):
    """
    Send the complete prompt to Antigravity through STDIN.

    IMPORTANT:
    The prompt is deliberately NOT passed through -p/argv.
    Hermes conversations can become very large.
    """

    prompt = build_prompt(request.messages)

    model = model or DEFAULT_MODEL

    command = [
        "/home/mehmet/.local/bin/agy",
        "--model",
        model,
        "--dangerously-skip-permissions",
    ]

    print(
        f"[bridge] Calling agy | "
        f"model={model!r} | "
        f"messages={len(request.messages)} | "
        f"prompt_chars={len(prompt):,}",
        flush=True,
    )

    started = time.time()

    try:
        result = subprocess.run(
            command,

            # Send the Hermes context through STDIN.
            input=prompt,

            capture_output=True,
            text=True,

            # Allow long agent/model calls.
            timeout=600,
        )

    except subprocess.TimeoutExpired:
        elapsed = time.time() - started

        print(
            f"[bridge] agy TIMEOUT after {elapsed:.1f}s",
            flush=True,
        )

        raise HTTPException(
            status_code=504,
            detail={
                "error": "Antigravity request timed out",
                "elapsed_seconds": round(elapsed, 2),
                "prompt_chars": len(prompt),
            },
        )

    elapsed = time.time() - started

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    print(
        f"[bridge] agy finished | "
        f"returncode={result.returncode} | "
        f"elapsed={elapsed:.1f}s | "
        f"stdout_chars={len(stdout):,} | "
        f"stderr_chars={len(stderr):,}",
        flush=True,
    )

    # agy explicitly reported failure.
    if result.returncode != 0:

        diagnostic = stderr.strip()

        if not diagnostic:
            diagnostic = stdout.strip()

        print(
            "[bridge] agy ERROR:\n"
            + (diagnostic[-4000:] if diagnostic else "(no diagnostic output)"),
            flush=True,
        )

        raise HTTPException(
            status_code=502,
            detail={
                "error": "Antigravity command failed",
                "returncode": result.returncode,
                "stderr": stderr.strip()[-4000:],
                "stdout": stdout.strip()[-2000:],
                "prompt_chars": len(prompt),
                "elapsed_seconds": round(elapsed, 2),
            },
        )

    answer = stdout.strip()

    # agy returned success but nothing usable.
    if not answer:

        print(
            "[bridge] agy returned EMPTY stdout\n"
            f"[bridge] stderr:\n"
            f"{stderr.strip()[-4000:] or '(empty)'}",
            flush=True,
        )

        raise HTTPException(
            status_code=502,
            detail={
                "error": "Antigravity returned an empty response",
                "returncode": result.returncode,
                "stderr": stderr.strip()[-4000:] or "(empty)",
                "prompt_chars": len(prompt),
                "elapsed_seconds": round(elapsed, 2),
            },
        )

    return answer


def call_agy(request):
    """Try the primary model, then the configured fallback on failure."""
    # The bridge owns model selection; callers may send a stale model name.
    primary = DEFAULT_MODEL
    models_to_try = [primary]
    if FALLBACK_MODEL not in models_to_try:
        models_to_try.append(FALLBACK_MODEL)

    last_error = None
    for index, model in enumerate(models_to_try):
        try:
            if index:
                print(
                    f"[bridge] Retrying with fallback model={model!r}",
                    flush=True,
                )
            return call_agy_once(request, model)
        except HTTPException as exc:
            last_error = exc
            if index == len(models_to_try) - 1:
                raise
            print(
                f"[bridge] model={model!r} failed; trying fallback",
                flush=True,
            )

    # The loop always returns or raises; keep a defensive error for static analyzers.
    raise last_error or HTTPException(status_code=502, detail="No model succeeded")


# ---------------------------------------------------------
# OpenAI-compatible model endpoints
# ---------------------------------------------------------

@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "google-antigravity",
            }
        ],
    }


@app.get("/v1/models/{model_name:path}")
def model_info(model_name: str):
    """
    Hermes probes the individual model endpoint.
    Previously this produced one of your harmless 404s.
    """

    return {
        "id": model_name,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "google-antigravity",
    }


# ---------------------------------------------------------
# OpenAI-compatible chat completions
# ---------------------------------------------------------

@app.post("/v1/chat/completions")
def chat(request: ChatRequest):

    answer = call_agy(request)

    completion_id = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())
    model = DEFAULT_MODEL

    # =====================================================
    # STREAMING
    # =====================================================

    if request.stream:

        def event_stream():

            # Initial role event.
            first_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant"
                        },
                        "finish_reason": None,
                    }
                ],
            }

            yield (
                "data: "
                + json.dumps(first_chunk, ensure_ascii=False)
                + "\n\n"
            )

            # agy has already completed its answer at this point.
            # Break it into OpenAI-compatible SSE chunks for Hermes.
            chunk_size = 100

            for i in range(0, len(answer), chunk_size):

                content_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": answer[i:i + chunk_size]
                            },
                            "finish_reason": None,
                        }
                    ],
                }

                yield (
                    "data: "
                    + json.dumps(
                        content_chunk,
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )

            # Hermes requires a terminal finish_reason.
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }

            yield (
                "data: "
                + json.dumps(final_chunk, ensure_ascii=False)
                + "\n\n"
            )

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # =====================================================
    # NON-STREAMING
    # =====================================================

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8765,
    )
