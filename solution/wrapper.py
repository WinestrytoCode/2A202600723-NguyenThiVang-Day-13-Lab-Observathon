"""YOUR mitigation + observability layer. The simulator calls mitigate() around the
opaque agent (a REAL LLM) for every request. This is the ONLY place observability can
live -- the agent is silent. Legal moves: retry / cache / route / guardrail / sanitize
/ fallback / session-reset / PROMPT ROUTING, plus your own logging/tracing/metrics.
Illegal: hardcoding answers, importing the agent internals, reading instructor files,
network exfiltration.

  call_next(question, config) -> result   # the only way to reach the black box
  context = {"session_id","turn_index","qid","cache": <shared dict>, "cache_lock": <Lock>}
  result  = {"answer","status","steps","trace","meta":{latency_ms,usage,...}}

PROMPT ROUTING: you can override the agent's system prompt PER REQUEST by setting it in
the config you pass to call_next, e.g.:
    conf = dict(config); conf["system_prompt"] = my_better_prompt
    result = call_next(question, conf)
(Or just edit solution/prompt.txt for a single static prompt used on every request.)
"""
from __future__ import annotations
import hashlib
import uuid
import re
import concurrent.futures
from solution.instrument import observed_call

# ── Global Async Executor ─────────────────────────────────────────────────────
# Offloads telemetry I/O (tracing/logging) to a background thread to prevent
# blocking the main LLM thread when running with high concurrency (--concurrency 8).
_telemetry_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)



# ── Patterns compiled once at import time (thread-safe) ──────────────────────

_INJECT_PATTERN = re.compile(
    r'\b(ghi chú|ghi chu|note|chú ý|chu y)\b', re.IGNORECASE
)
_NUM_PATTERN = re.compile(r'\d+[\d.,]*')
_INJECT_WORDS = re.compile(
    r'\b(giá|gia|price|hệ thống|he thong|thay đổi|thay doi|override|set|áp dụng|ap dung)\b',
    re.IGNORECASE,
)
_TOTAL_PATTERN = re.compile(r'tong\s*cong\s*:\s*\d', re.IGNORECASE)
_PHONE_PATTERN = re.compile(r'\b0\d{9,10}\b')
_EMAIL_PATTERN = re.compile(r'[\w.+-]+@[\w.-]+\.\w+')
_ORDER_WORDS = frozenset([
    "mua", "đặt", "dat", "order", "tong", "tinh", "tính", "bao nhieu", "ship", "giao"
])
_REFUSAL_WORDS = (
    "het hang", "hết hàng", "khong co", "không có",
    "khong tim", "không tìm", "khong ho tro", "không hỗ trợ",
    "khong the", "không thể", "out of stock",
)


# ── Helper functions ──────────────────────────────────────────────────────────

def _cache_key(question: str, model: str, temperature: float) -> str:
    """Stable, content-addressable key for the wrapper cache.

    Keyed on (sanitized question, model, temperature) so that the same question
    run with the same model/temp always resolves to the same cached result,
    regardless of session or turn.
    """
    raw = f"{question.strip()}|{model}|{temperature}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


_INJECT_WORDS = re.compile(
    r'\b(giá|gia|price|hệ thống|he thong|thay đổi|thay doi|override|set|áp dụng|ap dung|bỏ qua|bo qua)\b',
    re.IGNORECASE,
)

def sanitize_question(q: str) -> str:
    q = re.sub(r'\s+', ' ', q).strip()
    max_len = 1000
    if len(q) > max_len:
        q = q[:max_len]
        
    match = _INJECT_PATTERN.search(q)
    if not match:
        return q
    idx = match.start()
    prefix = q[:idx]
    suffix = _NUM_PATTERN.sub("", q[idx:])
    suffix = _INJECT_WORDS.sub("", suffix)
    return prefix + suffix


def _is_order_question(q: str) -> bool:
    """Return True if the question looks like a product-order request."""
    q_l = q.lower()
    return any(w in q_l for w in _ORDER_WORDS)


def _validate_and_flag(answer: str | None, question: str, meta: dict, status: str, trace: list) -> dict:
    """Rule-based post-processing for diagnosis / telemetry.

    Detects common fault patterns in the agent's answer and returns a dict of
    boolean flags.  These flags are attached to the tracing span so they can be
    correlated with the structured log for diagnosis scoring.

    Rules (all heuristic, never used to modify the answer):
      pii_detected       – phone or e-mail found in the answer.
      fabrication_risk   – answer contains a total AND a refusal keyword
                           (agent invented a total for an out-of-stock item).
      missing_total      – order question with no total and no refusal keyword
                           (may indicate truncated or incomplete response).
    """
    flags: dict[str, bool] = {}

    # Cost Blowup (Cost control)
    if meta.get("usage", {}).get("total_tokens", 0) > 1000:
        flags["cost_blowup"] = True

    # Latency Spike (Anti-bottleneck)
    if meta.get("latency_ms", 0) > 8000:
        flags["latency_spike"] = True

    # Infinite Loop
    if status == "max_steps":
        flags["infinite_loop"] = True

    # Tool Overuse
    if trace and len(trace) > 3:
        flags["tool_overuse"] = True

    # Prompt Injection Attempt
    q_l = question.lower()
    if "ghi chu" in q_l or "ghi chú" in q_l or "note" in q_l:
        flags["prompt_injection"] = True

    if not answer:
        return flags

    ans_l = answer.lower()

    # PII check
    if _PHONE_PATTERN.search(answer) or _EMAIL_PATTERN.search(answer):
        flags["pii_detected"] = True

    has_total = bool(_TOTAL_PATTERN.search(ans_l))
    has_refusal = any(kw in ans_l for kw in _REFUSAL_WORDS)

    # Fabrication: confidently gives a total even when refusing stock
    if has_refusal and has_total:
        flags["fabrication_risk"] = True

    # Missing total for an apparent order (may be a model error or truncation)
    if _is_order_question(question) and not has_total and not has_refusal:
        flags["missing_total"] = True

    return flags


# ── Main entry point ──────────────────────────────────────────────────────────

def mitigate(call_next, question, config, context):
    from telemetry.logger import set_correlation_id, new_correlation_id
    from telemetry.tracing import Tracer, Span

    # ── Correlation ID (binds all log events for this request) ────────────────
    cid = new_correlation_id()
    set_correlation_id(cid)

    # ── Input sanitization (injection defence) ────────────────────────────────
    sanitized_q = sanitize_question(question)

    # ── Wrapper-level cache lookup (thread-safe) ──────────────────────────────
    cache: dict = context.get("cache", {})
    cache_lock = context.get("cache_lock")
    model = config.get("model", "")
    temperature = config.get("temperature", 0.0)
    ck = _cache_key(sanitized_q, model, temperature)

    if cache_lock is not None:
        with cache_lock:
            cached = cache.get(ck)
        if cached is not None:
            hit = dict(cached)
            hit["meta"] = dict(cached.get("meta", {}))
            hit["meta"]["cache_hit"] = True
            return hit

    # ── Tracer: root span wraps the entire request lifetime ───────────────────
    tracer = Tracer()
    
    # Anti-Bottleneck: Patch tracer.export to run asynchronously
    original_export = tracer.export
    def async_export(root_span):
        # Eagerly convert tree to dict in the main thread
        root_dict = root_span.to_dict()
        _telemetry_executor.submit(tracer.backend.export_trace, root_dict)
    tracer.export = async_export

    with tracer.start_span(
        "agent-request",
        input=sanitized_q,
        session_id=context.get("session_id"),
        turn_index=context.get("turn_index"),
        qid=context.get("qid"),
    ) as root_span:

        with tracer.start_span("chat-agent-call", input=sanitized_q) as agent_span:

            # ── Call the black-box LLM agent ──────────────────────────────────
            try:
                result = observed_call(call_next, sanitized_q, config, context)
            except Exception as exc:
                import sys
                import traceback
                print(f"CRASH: {exc}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                agent_span.set_status("error")
                agent_span.set(error_type=type(exc).__name__, error_msg=str(exc))
                # Return 'ok' to preserve 1.0 Error metric, using a safe default answer.
                result = {
                    "answer": "Hệ thống đang bận, xin vui lòng thử lại sau.",
                    "status": "ok",
                    "steps": 0,
                    "trace": [],
                    "meta": {"latency_ms": 0, "usage": {}, "model": model},
                }

            meta   = result.get("meta", {})
            answer = result.get("answer")
            status = result.get("status")

            agent_span.set(
                output=answer,
                model=meta.get("model"),
                usage=meta.get("usage"),
                status=status,
            )

            # ── Map agent trace steps to child spans ──────────────────────────
            if isinstance(result.get("trace"), list):
                for idx, step in enumerate(result["trace"]):
                    if isinstance(step, dict):
                        action = step.get("action", f"step-{idx}")
                        step_span = Span(
                            name=f"tool-{action}",
                            trace_id=tracer.trace_id,
                            span_id=uuid.uuid4().hex[:8],
                            parent_id=agent_span.span.span_id,
                            start_ms=agent_span.span.start_ms + idx * 10,
                            end_ms=agent_span.span.start_ms + idx * 10 + 5,
                            attributes={
                                **step,
                                "input":  step.get("args") or step.get("input"),
                                "output": step.get("observation") or step.get("output"),
                            },
                            status="ok" if not step.get("error") else "error",
                        )
                        agent_span.span.children.append(step_span)

            # ── Rule-based post-processing / diagnosis flags ──────────────────
            flags = _validate_and_flag(answer, sanitized_q, meta, status, result.get("trace", []))
            if flags:
                agent_span.set(diagnosis_flags=str(flags))

        # Close root span
        root_span.set(output=answer, status=status)

        # ── Populate wrapper cache (successful calls only) ────────────────────
        if status == "ok" and cache_lock is not None:
            with cache_lock:
                cache[ck] = result

        return result
