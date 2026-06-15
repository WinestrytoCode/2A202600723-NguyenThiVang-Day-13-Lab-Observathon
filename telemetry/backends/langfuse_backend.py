"""Send traces to Langfuse (Track 4, real backend) -- activates only with keys.

Uses the CURRENT Langfuse Python SDK v4 (2026), which is OpenTelemetry-based.
Requires `pip install langfuse` and env LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY
(+ optional LANGFUSE_HOST for self-host). If the SDK or keys are missing, the
constructor raises so the factory falls back to the file backend -- the zero-key
path is never broken by selecting langfuse without keys.
"""
from __future__ import annotations
import os
from telemetry.backends.base import Backend


class LangfuseBackend(Backend):
    def __init__(self):
        if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
            raise RuntimeError("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set")
        try:
            from langfuse import get_client  # langfuse v4 SDK
        except ImportError as exc:
            raise RuntimeError("langfuse not installed (pip install langfuse)") from exc
        self._client = get_client()

    def _sanitize_metadata(self, attributes: dict) -> dict[str, str]:
        sanitized = {}
        for k, v in attributes.items():
            if v is None:
                continue
            # Ensure metadata values are strings of max length 200
            v_str = str(v)
            if len(v_str) > 200:
                v_str = v_str[:197] + "..."
            sanitized[str(k)] = v_str
        return sanitized

    def _emit(self, span: dict, parent=None):
        name = span["name"]
        attrs = span.get("attributes", {}).copy()

        # Determine observation type
        as_type = "generation" if name.startswith("chat") or name.startswith("agent") else "span"

        # Extract inputs/outputs/model/usage if present
        obs_input = attrs.pop("input", None)
        obs_output = attrs.pop("output", None)
        obs_model = attrs.pop("model", None)
        obs_usage = attrs.pop("usage", None)

        # Sanitize remaining attributes as metadata
        metadata = self._sanitize_metadata({
            **attrs,
            "duration_ms": span["duration_ms"],
            "status": span["status"]
        })

        # Start the observation
        cm = self._client.start_as_current_observation(
            name=name,
            as_type=as_type,
            input=obs_input,
            model=obs_model,
            metadata=metadata,
        )
        with cm as obs:
            if obs_output is not None or obs_usage is not None:
                # Format usage dictionary for Langfuse v4
                formatted_usage = None
                if obs_usage:
                    # Our app's usage is: {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
                    prompt_tokens = obs_usage.get("prompt_tokens", 0)
                    completion_tokens = obs_usage.get("completion_tokens", 0)
                    total_tokens = obs_usage.get("total_tokens", prompt_tokens + completion_tokens)
                    formatted_usage = {
                        "input": prompt_tokens,
                        "output": completion_tokens,
                        "total": total_tokens
                    }

                obs.update(output=obs_output, usage=formatted_usage)

            for child in span.get("children", []):
                self._emit(child)

    def export_trace(self, trace: dict) -> None:
        attrs = trace.get("attributes", {})
        session_id = str(attrs.get("session_id")) if attrs.get("session_id") is not None else None
        user_id = str(attrs.get("user_id")) if attrs.get("user_id") is not None else None

        from langfuse import propagate_attributes
        with propagate_attributes(session_id=session_id, user_id=user_id):
            self._emit(trace)
        self._client.flush()
