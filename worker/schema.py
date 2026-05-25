"""Input schema (rp_validator) + cross-field validation."""

from __future__ import annotations

from typing import Any

from runpod.serverless.utils.rp_validator import validate


VALID_RETURNS = {"tarball_b64", "inline", "s3"}

# MinerU 3.1.x backends. Validated at the handler boundary so callers get a
# friendly error instead of a deep MinerU stack trace.
VALID_BACKENDS = {
    "pipeline",
    "vlm-auto-engine",
    "vlm-http-client",
    "hybrid-auto-engine",
    "hybrid-http-client",
}


# rp_validator's `constraints` lambdas are silently ignored on some versions
# — we declare them anyway for documentation but never rely on them.
# Cross-field rules and per-field bounds are re-checked manually in
# validate_input() below.
INPUT_SCHEMA: dict[str, dict[str, Any]] = {
    "file_url":       {"type": str,  "required": False, "default": None},
    "file_b64":       {"type": str,  "required": False, "default": None},
    "volume_path":    {"type": str,  "required": False, "default": None},
    # When `probe` is true the handler skips MinerU entirely and returns a
    # filesystem dump of /runpod-volume + relevant env vars. Used to debug
    # RunPod Cached Models setup.
    "probe":          {"type": bool, "required": False, "default": False},
    "start_page":     {"type": int,  "required": False, "default": 0},
    "end_page":       {"type": int,  "required": False, "default": -1},
    "lang":           {"type": str,  "required": False, "default": "en"},
    "backend":        {"type": str,  "required": False, "default": "vlm-auto-engine"},
    "server_url":     {"type": str,  "required": False, "default": None},
    "formula_enable": {"type": bool, "required": False, "default": True},
    "table_enable":   {"type": bool, "required": False, "default": True},
    "return":         {"type": str,  "required": False, "default": "tarball_b64"},
    "basename":       {"type": str,  "required": False, "default": "doc"},
}


def _fail(msg: str) -> None:
    raise ValueError(f"input validation failed: {msg}")


def validate_input(job_input: dict) -> dict:
    """Run rp_validator over the schema and enforce cross-field rules.

    Returns the cleaned input dict with defaults applied. Raises ValueError
    with an ``input validation failed: ...`` prefix on any rejection.
    """
    result = validate(job_input, INPUT_SCHEMA)
    if result.get("errors"):
        _fail("; ".join(result["errors"]))

    cleaned = result["validated_input"]

    basename = cleaned.get("basename") or "doc"
    if not basename or not all(c.isalnum() or c in "-_" for c in basename):
        _fail(f"basename must be alphanumeric (with - or _); got {basename!r}")

    ret = cleaned.get("return") or "tarball_b64"
    if ret not in VALID_RETURNS:
        _fail(f"return must be one of {sorted(VALID_RETURNS)}; got {ret!r}")

    backend = cleaned.get("backend") or "vlm-auto-engine"
    if backend not in VALID_BACKENDS:
        _fail(f"backend must be one of {sorted(VALID_BACKENDS)}; got {backend!r}")

    start_page = cleaned.get("start_page", 0) or 0
    if start_page < 0:
        _fail(f"start_page must be >= 0; got {start_page!r}")

    # XOR over the three transports. The handler also relies on this — only
    # one of file_url/file_b64/volume_path may be set per job.
    sources = [k for k in ("file_url", "file_b64", "volume_path") if cleaned.get(k)]
    if len(sources) != 1:
        _fail(
            f"must provide exactly one of file_url / file_b64 / volume_path "
            f"(got {sources!r})"
        )

    if backend.endswith("-http-client") and not cleaned.get("server_url"):
        _fail(
            f"backend={backend!r} requires `server_url` pointing at an "
            f"external vLLM OpenAI-compatible server"
        )

    return cleaned
