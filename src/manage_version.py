"""Update sharkeye-app-build/latest_version.json with the commit SHA + timestamp for a build platform."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import storage

BUCKET_NAME = "sharkeye-app-build"
VERSION_BLOB_NAME = "latest_version.json"

PLATFORMS = ("windows", "macos_intel", "macos_silicon")
MAX_RETRIES = 5


def _empty_entry() -> dict[str, str]:
    return {"latest_commit": "", "committed_at": ""}


def _empty_versions() -> dict[str, dict[str, str]]:
    return {platform: _empty_entry() for platform in PLATFORMS}


def _coerce_entry(value: object) -> dict[str, str]:
    """Normalize a stored value into a {latest_commit, committed_at} entry.

    Accepts the legacy formats where the value was a bare SHA string or used a
    "commit" key instead of "latest_commit" (the key the Cloud Function reads).
    """
    if isinstance(value, dict):
        commit = value.get("latest_commit", value.get("commit", ""))
        committed_at = value.get("committed_at", "")
        return {
            "latest_commit": commit if isinstance(commit, str) else str(commit),
            "committed_at": committed_at if isinstance(committed_at, str) else str(committed_at),
        }
    if isinstance(value, str):
        return {"latest_commit": value, "committed_at": ""}
    return _empty_entry()


def get_commit_timestamp(commit: str) -> str:
    """Return the committer date of ``commit`` as an ISO-8601 string, or '' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "show", "-s", "--format=%cI", commit],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"Could not resolve commit timestamp from git: {exc}")
        return ""


def _parse_versions(data: object) -> dict[str, dict[str, str]]:
    versions = _empty_versions()
    if isinstance(data, dict):
        for platform in PLATFORMS:
            if platform in data:
                versions[platform] = _coerce_entry(data[platform])
    return versions


def load_versions(bucket: storage.Bucket) -> tuple[dict[str, dict[str, str]], int]:
    """Return (versions, generation). generation is 0 if the blob does not exist.

    Download the *live* object on a generation-free Blob. Reusing a Blob after
    reload() pins ``blob.generation``; Blob.exists() then queries that specific
    generation. After a concurrent overwrite the old generation is gone, exists()
    returns False, and a create with ifGenerationMatch=0 412s on every retry.
    """
    blob = bucket.blob(VERSION_BLOB_NAME)
    try:
        data = json.loads(blob.download_as_text())
    except NotFound:
        return _empty_versions(), 0
    generation = blob.generation
    return _parse_versions(data), 0 if generation is None else int(generation)


def update_latest_version(platform: str, commit: str, committed_at: str) -> dict[str, dict[str, str]]:
    if platform not in PLATFORMS:
        raise ValueError(f"platform must be one of {PLATFORMS}, got {platform!r}")
    if not commit:
        raise ValueError("commit SHA must be non-empty")

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    for attempt in range(1, MAX_RETRIES + 1):
        versions, generation = load_versions(bucket)
        versions[platform] = {"latest_commit": commit, "committed_at": committed_at}
        payload = json.dumps(versions, indent=2, sort_keys=True) + "\n"

        # Fresh Blob so a prior generation is never attached as a query param.
        blob = bucket.blob(VERSION_BLOB_NAME)
        try:
            # Optimistic concurrency so parallel platform builds don't clobber each other.
            # generation=0 is GCS's "create only if the object does not exist" sentinel.
            blob.upload_from_string(
                payload,
                content_type="application/json",
                if_generation_match=generation,
            )
            print(f"Updated {VERSION_BLOB_NAME}: {platform} -> {commit} ({committed_at})")
            print(json.dumps(versions, indent=2, sort_keys=True))
            return versions
        except PreconditionFailed:
            if attempt == MAX_RETRIES:
                raise
            print(
                f"Concurrent update detected (attempt {attempt}/{MAX_RETRIES}); retrying..."
            )
            time.sleep(0.5 * attempt)

    raise RuntimeError("Failed to update latest_version.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record the latest build commit SHA + timestamp in GCS latest_version.json."
    )
    parser.add_argument(
        "--platform",
        type=str,
        required=True,
        choices=PLATFORMS,
        help="Build platform key to update",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default=None,
        help="Commit SHA (defaults to GITHUB_SHA)",
    )
    parser.add_argument(
        "--committed-at",
        type=str,
        default=None,
        help="ISO-8601 commit timestamp (defaults to git committer date of the commit)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commit = args.commit or os.environ.get("GITHUB_SHA", "")
    if not commit:
        raise SystemExit(
            "No commit SHA provided. Pass --commit or set GITHUB_SHA."
        )
    committed_at = args.committed_at or get_commit_timestamp(commit)
    update_latest_version(args.platform, commit, committed_at)


if __name__ == "__main__":
    main()
