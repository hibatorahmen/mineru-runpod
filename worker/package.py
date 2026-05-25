"""Three ways to ship MinerU's output back to the caller.

- tarball_b64: base64-encoded gzip-tar embedded in the response
- inline:      markdown + content_list + middle + images embedded directly
- s3:          tarball uploaded to an S3-compatible bucket, presigned URL returned
"""

from __future__ import annotations

import base64
import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any


def _build_tarball_bytes(output_dir: Path) -> bytes:
    """Gzip-tar the MinerU output dir; returns the raw bytes."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for child in sorted(output_dir.iterdir()):
            tar.add(child, arcname=child.name, recursive=True)
    return buf.getvalue()


def package_tarball(output_dir: Path) -> str:
    """Same archive as _build_tarball_bytes, base64-encoded for JSON transport."""
    return base64.b64encode(_build_tarball_bytes(output_dir)).decode("ascii")


def package_inline(output_dir: Path, basename: str) -> dict[str, Any]:
    md_path = output_dir / f"{basename}.md"
    cl_path = output_dir / f"{basename}_content_list.json"
    if not cl_path.is_file():
        cl_path = output_dir / f"{basename}_content_list_v2.json"
    mid_path = output_dir / f"{basename}_middle.json"

    images: dict[str, str] = {}
    images_dir = output_dir / "images"
    if images_dir.is_dir():
        for img in sorted(images_dir.iterdir()):
            if img.is_file():
                images[img.name] = base64.b64encode(img.read_bytes()).decode("ascii")

    return {
        "markdown": md_path.read_text(encoding="utf-8") if md_path.is_file() else "",
        "content_list": json.loads(cl_path.read_text(encoding="utf-8")) if cl_path.is_file() else [],
        "middle": json.loads(mid_path.read_text(encoding="utf-8")) if mid_path.is_file() else {},
        "images": images,
    }


# Default presigned URL lifetime for `return: "s3"` uploads.
# An hour is enough for a caller to fetch the tarball but short enough that a
# leaked URL stops working before it's interesting.
S3_PRESIGN_TTL_SECONDS = 3600


def package_s3(output_dir: Path, basename: str) -> dict[str, Any]:
    """Upload the output tarball to an S3-compatible bucket and return a
    presigned GET URL.

    Required worker env vars: BUCKET_ENDPOINT_URL, BUCKET_NAME,
    BUCKET_ACCESS_KEY_ID, BUCKET_SECRET_ACCESS_KEY. Optional:
    BUCKET_REGION (some providers need this; default empty), BUCKET_PREFIX
    (key path prefix inside the bucket; default empty).
    """
    endpoint = os.environ.get("BUCKET_ENDPOINT_URL", "").strip()
    bucket = os.environ.get("BUCKET_NAME", "").strip()
    access_key = os.environ.get("BUCKET_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("BUCKET_SECRET_ACCESS_KEY", "").strip()
    missing = [
        name for name, val in (
            ("BUCKET_ENDPOINT_URL", endpoint),
            ("BUCKET_NAME", bucket),
            ("BUCKET_ACCESS_KEY_ID", access_key),
            ("BUCKET_SECRET_ACCESS_KEY", secret_key),
        ) if not val
    ]
    if missing:
        raise ValueError(
            f"return='s3' requires worker env vars: {', '.join(missing)}. "
            f"Set these in the RunPod endpoint env config and redeploy."
        )

    region = os.environ.get("BUCKET_REGION", "").strip() or None
    prefix = os.environ.get("BUCKET_PREFIX", "").strip().lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    # boto3 import is lazy so workers that never call return='s3' don't pay
    # the ~50 MB cold-import cost.
    import boto3  # noqa: PLC0415
    from botocore.client import Config  # noqa: PLC0415

    tarball_bytes = _build_tarball_bytes(output_dir)
    # Use a UUID so concurrent jobs with the same basename don't collide.
    import uuid  # noqa: PLC0415
    key = f"{prefix}{basename}-{uuid.uuid4().hex}.tar.gz"

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        # SigV4 is required by most S3-compatible providers (R2, B2, MinIO).
        config=Config(signature_version="s3v4"),
    )
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=tarball_bytes,
        ContentType="application/gzip",
    )
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=S3_PRESIGN_TTL_SECONDS,
    )
    return {
        "tarball_url": url,
        "tarball_url_expires_in": S3_PRESIGN_TTL_SECONDS,
        "bucket_key": key,
        "bucket_bytes": len(tarball_bytes),
    }
