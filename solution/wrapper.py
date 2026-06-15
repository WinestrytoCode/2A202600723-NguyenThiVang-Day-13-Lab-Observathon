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
import uuid
import re
from solution.instrument import observed_call


def sanitize_question(q: str) -> str:
    # Spot order notes / instructions (e.g. "Ghi chú", "Note", "Chú ý") and strip prompt injection overrides
    pattern = re.compile(r'\b(ghi chú|ghi chu|note|chú ý|chu y)\b', re.IGNORECASE)
    match = pattern.search(q)
    if match:
        idx = match.start()
        prefix = q[:idx]
        suffix = q[idx:]
        # Neutralize any numeric prices in the note section to block price injection
        suffix = re.sub(r'\d+[\d.,]*', '', suffix)
        # Strip common instruction keywords
        for word in ["giá", "gia", "price", "hệ thống", "he thong", "thay đổi", "thay doi", "override", "set", "áp dụng", "ap dung"]:
            suffix = re.compile(re.escape(word), re.IGNORECASE).sub("", suffix)
        return prefix + suffix
    return q


def mitigate(call_next, question, config, context):
    from telemetry.logger import set_correlation_id, new_correlation_id
    from telemetry.tracing import Tracer, Span

    # Set correlation ID for structured logging correlation
    cid = new_correlation_id()
    set_correlation_id(cid)

    # Sanitize prompt injection attempts in the question notes
    sanitized_q = sanitize_question(question)

    # Initialize Tracer (which uses factory to resolve Langfuse or Console/File backends)
    tracer = Tracer()
    with tracer.start_span("agent-request", input=sanitized_q, session_id=context.get("session_id"), turn_index=context.get("turn_index"), qid=context.get("qid")) as root_span:
        with tracer.start_span("chat-agent-call", input=sanitized_q) as agent_span:
            # Invoke the observed black-box agent with sanitized input
            result = observed_call(call_next, sanitized_q, config, context)
            
            meta = result.get("meta", {})
            agent_span.set(
                output=result.get("answer"),
                model=meta.get("model"),
                usage=meta.get("usage"),
                status=result.get("status")
            )
            
            # Map the agent's internal trace steps to child spans under the agent-call
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
                                "input": step.get("args") or step.get("input"),
                                "output": step.get("observation") or step.get("output")
                            },
                            status="ok" if not step.get("error") else "error"
                        )
                        agent_span.span.children.append(step_span)

        root_span.set(
            output=result.get("answer"),
            status=result.get("status")
        )
        return result
