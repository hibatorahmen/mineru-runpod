"""RunPod serverless entry point for the MinerU worker.

The pieces this orchestrates live in the worker/ package:
  worker.schema   — input validation
  worker.io       — fetch raw bytes from URL / b64 / volume + format detection
  worker.parse    — MinerU lazy import + async parse call
  worker.package  — tarball / inline / s3 response packaging
  worker.debug    — GPU info, model dir, /runpod-volume probe

The module surface (``handler.MAX_INLINE_FILE_MB``, ``handler._detect_format``,
``handler._validate_input``, ``handler._package_tarball``, etc.) is preserved
for tests/back-compat — see the re-exports near the bottom of this file.
"""

from __future__ import annotations

import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import runpod

from worker import debug as _debug
from worker import io as _io
from worker import package as _package
from worker import parse as _parse
from worker import schema as _schema


def _maybe_progress(job: dict, data: dict) -> None:
    """Best-effort progress update. Tests / sync clients without a job id
    shouldn't fail just because we tried to surface progress."""
    try:
        runpod.serverless.progress_update(job, data)
    except Exception as e:  # noqa: BLE001
        # Leave a breadcrumb in worker logs so a silent failure here is at
        # least visible during incident debugging.
        print(f"[mineru-worker] progress_update failed: {e!r}", flush=True)


def _build_debug(phase_ms: dict[str, int], gpu_info: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "gpu": gpu_info,
        "model_dir": _debug.find_model_dir(),
        "phase_ms": phase_ms,
        **extra,
    }


async def _handle_probe(started: float, gpu_info: dict[str, Any], phase_ms: dict[str, int]) -> dict[str, Any]:
    print("[mineru-worker] probe job: dumping filesystem layout", flush=True)
    return {
        "ok": True,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "mineru_version": _parse.MINERU_VERSION,
        "mineru_available": _parse.MINERU_AVAILABLE,
        "probe": _debug.probe_filesystem(),
        "debug": _build_debug(phase_ms, gpu_info),
    }


async def _handle_parse(
    job: dict,
    cleaned: dict[str, Any],
    started: float,
    gpu_info: dict[str, Any],
    phase_ms: dict[str, int],
) -> dict[str, Any]:
    # rp_validator's strict typing forces end_page to be an int; translate
    # the -1 sentinel back to None so MinerU treats it as "until end of doc".
    end_page_val = cleaned["end_page"]
    end_page = None if end_page_val is None or end_page_val < 0 else int(end_page_val)
    backend = cleaned["backend"]

    print(
        f"[mineru-worker] starting job: backend={backend} lang={cleaned['lang']} "
        f"start={cleaned['start_page']} end={end_page} "
        f"gpu={gpu_info.get('name', '?')} cc={gpu_info.get('compute_capability', '?')}",
        flush=True,
    )

    _maybe_progress(job, {"phase": "fetching_input"})
    t = time.monotonic()
    file_bytes, source = await _io.resolve_input_bytes(cleaned)
    phase_ms["fetch_input"] = int((time.monotonic() - t) * 1000)

    input_format = _io.detect_format(file_bytes)
    if input_format == "unknown":
        raise ValueError(
            "input bytes do not match any supported format "
            "(PDF, PNG/JPEG/GIF/BMP/TIFF/WebP image, or DOCX/PPTX/XLSX). "
            "Check that file_b64 was base64-encoded correctly and that "
            "file_url returned the file body (not an error page)."
        )

    _maybe_progress(job, {
        "phase": "parsing",
        "input_bytes": len(file_bytes),
        "input_format": input_format,
        "start_page": cleaned["start_page"],
        "end_page": end_page,
    })

    with tempfile.TemporaryDirectory(prefix="mineru-job-") as tmp:
        work_dir = Path(tmp)
        t = time.monotonic()
        output_dir = await _parse.run_mineru(
            file_bytes,
            basename=cleaned["basename"],
            work_dir=work_dir,
            input_format=input_format,
            start_page=cleaned["start_page"],
            end_page=end_page,
            lang=cleaned["lang"],
            backend=backend,
            server_url=cleaned.get("server_url"),
            formula_enable=cleaned["formula_enable"],
            table_enable=cleaned["table_enable"],
        )
        phase_ms["mineru_parse"] = int((time.monotonic() - t) * 1000)

        _maybe_progress(job, {"phase": "packaging"})

        t = time.monotonic()
        # `pages_requested` reflects the slice the caller asked for, NOT the
        # number MinerU actually produced (MinerU may emit fewer if the doc
        # is shorter than end_page). -1 == "full document".
        pages_requested = (
            (end_page - cleaned["start_page"] + 1) if end_page is not None else -1
        )
        response: dict[str, Any] = {
            "ok": True,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "pages_requested": pages_requested,
            # Back-compat: older clients read `pages_processed`. Same value.
            "pages_processed": pages_requested,
            "mineru_version": _parse.MINERU_VERSION,
            "source": source,
        }
        if cleaned["return"] == "inline":
            response.update(_package.package_inline(output_dir, cleaned["basename"]))
        elif cleaned["return"] == "s3":
            response.update(_package.package_s3(output_dir, cleaned["basename"]))
        else:
            response["tarball_b64"] = _package.package_tarball(output_dir)
        phase_ms["package"] = int((time.monotonic() - t) * 1000)

        response["debug"] = _build_debug(
            phase_ms, gpu_info, backend=backend, input_format=input_format
        )
        print(
            f"[mineru-worker] done: elapsed={response['elapsed_seconds']}s "
            f"phase_ms={phase_ms} model_dir={response['debug']['model_dir']}",
            flush=True,
        )
        return response


async def handler(job: dict) -> dict:
    started = time.monotonic()
    phase_ms: dict[str, int] = {}
    gpu_info = _debug.collect_gpu_info()
    try:
        raw_input = job.get("input") or {}
        # Probe mode bypasses schema validation: a probe has no file source
        # and the operator may want to send arbitrary debug flags through.
        if raw_input.get("probe") is True:
            return await _handle_probe(started, gpu_info, phase_ms)

        cleaned = _schema.validate_input(raw_input)
        return await _handle_parse(job, cleaned, started, gpu_info, phase_ms)

    except Exception as exc:  # noqa: BLE001
        # Top-level `error` key tells RunPod to mark this job FAILED.
        # Keep `ok=false` and the structured details so clients see context.
        print(f"[mineru-worker] failed: {type(exc).__name__}: {exc}", flush=True)
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "ok": False,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "mineru_version": _parse.MINERU_VERSION,
            "traceback": traceback.format_exc(limit=5),
            "debug": _build_debug(phase_ms, gpu_info),
        }


# -----------------------------------------------------------------------------
# Back-compat surface for tests and any out-of-tree callers that imported
# helpers from this module directly. New code should import from worker.*.
# -----------------------------------------------------------------------------

MAX_INLINE_FILE_MB = _io.MAX_INLINE_FILE_MB
MINERU_VERSION = _parse.MINERU_VERSION
_MINERU_AVAILABLE = _parse.MINERU_AVAILABLE

_resolve_input_bytes = _io.resolve_input_bytes
_detect_format = _io.detect_format
_validate_input = _schema.validate_input
_package_tarball = _package.package_tarball
_package_inline = _package.package_inline
_package_s3 = _package.package_s3
_build_tarball_bytes = _package._build_tarball_bytes
_run_mineru = _parse.run_mineru
_collect_gpu_info = _debug.collect_gpu_info
_find_model_dir = _debug.find_model_dir
_probe_filesystem = _debug.probe_filesystem


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
