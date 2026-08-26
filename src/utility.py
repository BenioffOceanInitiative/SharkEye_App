import os
import sys
import json


def select_torch_device():
    """Prefer CUDA, then MPS, then CPU.

    GitHub Actions macOS runners advertise MPS but typically cannot allocate
    shared GPU memory, so skip MPS when CI=true.
    """
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if os.environ.get("CI", "").lower() in ("1", "true", "yes"):
        return torch.device("cpu")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        return torch.device("mps")
    return torch.device("cpu")


def get_base_dir():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


def get_user_data_dir():
    """Per-user writable data root (experiments, logs, synced docs).

    Used when the frozen app lives in a non-writable location such as
    ``C:\\Program Files\\SharkEye``.
    """
    if sys.platform.startswith("win"):
        root = os.environ.get("LOCALAPPDATA") or os.path.join(
            os.path.expanduser("~"), "AppData", "Local"
        )
        return os.path.join(root, "SharkEye")
    if sys.platform == "darwin":
        return os.path.join(
            os.path.expanduser("~"), "Library", "Application Support", "SharkEye"
        )
    xdg = os.environ.get("XDG_DATA_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "share"
    )
    return os.path.join(xdg, "SharkEye")


def _path_is_under(path, root):
    try:
        path_abs = os.path.abspath(path)
        root_abs = os.path.abspath(root)
        return os.path.commonpath([path_abs, root_abs]) == root_abs
    except ValueError:
        return False


def _frozen_exe_is_in_program_files():
    if not sys.platform.startswith("win"):
        return False
    exe_dir = os.path.dirname(sys.executable)
    for key in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
        root = os.environ.get(key)
        if root and _path_is_under(exe_dir, root):
            return True
    return False


def _ensure_writable_dir(path):
    """Create *path* and verify this process can write into it."""
    os.makedirs(path, exist_ok=True)
    probe = os.path.join(path, ".sharkeye_write_test")
    with open(probe, "w", encoding="utf-8") as handle:
        handle.write("ok")
    os.remove(probe)


def get_results_dir():
    if getattr(sys, 'frozen', False):
        candidates = [os.path.join(os.path.dirname(sys.executable), "results")]
        # An Inno/MSI install under Program Files is not writable without
        # elevation; skip it so startup does not PermissionError.
        if _frozen_exe_is_in_program_files():
            candidates = []
        candidates.append(os.path.join(get_user_data_dir(), "results"))
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(base_dir, '..'))
        candidates = [os.path.join(base_dir, "results")]

    last_error = None
    for results_dir in candidates:
        try:
            _ensure_writable_dir(results_dir)
            return results_dir
        except OSError as exc:
            last_error = exc
    raise PermissionError(
        f"Cannot create a writable results directory. Last error: {last_error}"
    ) from last_error


def resource_path(relative_path):
    """ Get the absolute path to a resource, works for dev and PyInstaller. """
    if getattr(sys, 'frozen', False):
        # Running in a PyInstaller bundle
        base_path = sys._MEIPASS
        # Check if we're in a Mac app bundle
        if sys.platform == 'darwin' and '.app' in base_path:
            # Navigate to the Resources folder in the Mac app bundle
            base_path = os.path.abspath(os.path.join(base_path, '..', 'Resources'))
    else:
        # Running in a normal Python environment
        base_path = os.path.dirname(os.path.abspath(__file__))
        # Move back one directory
        base_path = os.path.abspath(os.path.join(base_path, '..'))
    
    return os.path.join(base_path, relative_path)


def get_bundled_docs_dir():
    """Return the read-only (or repo) docs directory shipped with the app."""
    return resource_path("docs")


def get_writable_docs_dir():
    """User-writable docs cache for synced help content.

    Frozen builds cannot reliably overwrite bundled docs under _MEIPASS / Mac
    Resources, so updates land beside results/ (same writable base as experiments).
    Dev runs use the repo docs/ folder directly.
    """
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(get_results_dir())
        docs_dir = os.path.join(base_dir, "docs")
    else:
        docs_dir = get_bundled_docs_dir()
    os.makedirs(docs_dir, exist_ok=True)
    return docs_dir


def _docs_dir_is_complete(docs_dir):
    """True when the guide, version stamp, and at least one image are present."""
    if not docs_dir:
        return False
    guide = os.path.join(docs_dir, "USER_GUIDE_VISUAL.md")
    version = os.path.join(docs_dir, "docs_version.json")
    images_dir = os.path.join(docs_dir, "images")
    if not (os.path.isfile(guide) and os.path.isfile(version) and os.path.isdir(images_dir)):
        return False
    try:
        return any(os.path.isfile(os.path.join(images_dir, name)) for name in os.listdir(images_dir))
    except OSError:
        return False


def local_help_docs_present(docs_dir=None):
    """True when docs/USER_GUIDE_VISUAL.md and docs/images/ (with files) exist."""
    docs_dir = docs_dir or get_writable_docs_dir()
    guide = os.path.join(docs_dir, "USER_GUIDE_VISUAL.md")
    images_dir = os.path.join(docs_dir, "images")
    if not os.path.isfile(guide) or not os.path.isdir(images_dir):
        return False
    try:
        return any(os.path.isfile(os.path.join(images_dir, name)) for name in os.listdir(images_dir))
    except OSError:
        return False



def resolve_help_docs_dir():
    """Prefer the writable/synced docs cache when complete; else bundled docs."""
    writable = get_writable_docs_dir()
    if _docs_dir_is_complete(writable):
        return writable
    bundled = get_bundled_docs_dir()
    if _docs_dir_is_complete(bundled):
        return bundled
    # Fall back to whichever exists so callers can surface a clear missing-file error.
    if os.path.isdir(writable):
        return writable
    return bundled


def resolve_help_guide_path():
    """Absolute path to USER_GUIDE_VISUAL.md from the resolved docs directory."""
    return os.path.join(resolve_help_docs_dir(), "USER_GUIDE_VISUAL.md")


def read_local_doc_version(docs_dir=None):
    """Read doc_version from docs_version.json; return 0 if missing/invalid."""
    docs_dir = docs_dir or resolve_help_docs_dir()
    version_path = os.path.join(docs_dir, "docs_version.json")
    try:
        with open(version_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("doc_version", 0))
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0


def write_local_doc_version(doc_version, docs_dir=None):
    """Write docs_version.json into the writable docs directory."""
    docs_dir = docs_dir or get_writable_docs_dir()
    os.makedirs(docs_dir, exist_ok=True)
    version_path = os.path.join(docs_dir, "docs_version.json")
    with open(version_path, "w", encoding="utf-8") as f:
        json.dump({"doc_version": int(doc_version)}, f, indent=2)
        f.write("\n")


def _split_gcs_path(gcs_path: str) -> tuple[str, str]:
    """Split ``bucket/blob`` or ``gs://bucket/blob`` into (bucket, blob_path)."""
    path = gcs_path.strip().replace("\\", "/")
    if path.startswith("gs://"):
        path = path[len("gs://"):]
    path = path.lstrip("/")
    if "/" not in path:
        raise ValueError(f"GCS path must include a blob path after the bucket: {gcs_path!r}")
    bucket_name, blob_path = path.split("/", 1)
    return bucket_name, blob_path.rstrip("/")


def _resolve_local_path(path: str) -> str:
    """Resolve a local path; relative paths are rooted at the project/docs base."""
    if os.path.isabs(path):
        return path
    # Prefer repo-root resolution (same base as resource_path("docs/...")).
    candidate = resource_path(path)
    if os.path.exists(candidate):
        return candidate
    return os.path.abspath(path)


def post_help_docs(
    input_images: str = "docs/images",
    input_doc: str = "docs/USER_GUIDE_VISUAL.md",
    output_images: str = "sharkeye-app-build/help_docs/images",
    output_doc: str = "sharkeye-app-build/help_docs/USER_GUIDE_VISUAL.md",
    archive: str = "sharkeye-app-build/archive/help_docs",
    docs_latest_version: str = "sharkeye-app-build/docs_latest_version.json",
) -> int:
    """Upload local help docs to GCS, archiving any existing objects first.

    Existing blobs at ``output_doc`` / under ``output_images`` are copied into a
    timestamped folder under ``archive``, then deleted. After upload, reads
    ``docs_latest_version``, increments ``doc_version`` by 1, and writes it back.

    Returns the new ``doc_version``.
    """
    from datetime import datetime
    from pathlib import Path

    from google.cloud import storage

    images_dir = Path(_resolve_local_path(input_images))
    doc_path = Path(_resolve_local_path(input_doc))
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not doc_path.is_file():
        raise FileNotFoundError(f"Markdown guide not found: {doc_path}")

    image_files = [p for p in images_dir.iterdir() if p.is_file()]
    if not image_files:
        raise FileNotFoundError(f"No image files in {images_dir}")

    out_images_bucket, out_images_prefix = _split_gcs_path(output_images)
    out_doc_bucket, out_doc_blob = _split_gcs_path(output_doc)
    archive_bucket, archive_prefix = _split_gcs_path(archive)
    version_bucket, version_blob_name = _split_gcs_path(docs_latest_version)

    buckets = {out_images_bucket, out_doc_bucket, archive_bucket, version_bucket}
    if len(buckets) != 1:
        raise ValueError(
            f"All GCS destinations must share one bucket; got {sorted(buckets)}"
        )
    bucket_name = out_images_bucket

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Collect existing help-doc blobs to archive (markdown + everything under images/).
    to_archive: list = []
    doc_blob = bucket.blob(out_doc_blob)
    if doc_blob.exists():
        to_archive.append(doc_blob)

    images_prefix = out_images_prefix.rstrip("/") + "/"
    for blob in client.list_blobs(bucket_name, prefix=images_prefix):
        if blob.name.endswith("/"):
            continue
        to_archive.append(blob)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = []
    for blob in to_archive:
        # Preserve path relative to help_docs/ when possible; else use basename.
        name = blob.name
        help_docs_marker = "help_docs/"
        if help_docs_marker in name:
            relative = name.split(help_docs_marker, 1)[1]
        else:
            relative = os.path.basename(name)
        archive_name = f"{archive_prefix}/{stamp}/{relative}"
        bucket.copy_blob(blob, bucket, archive_name)
        blob.delete()
        archived.append(archive_name)

    # Upload markdown.
    bucket.blob(out_doc_blob).upload_from_filename(str(doc_path))

    # Upload images (flat: local filename -> output_images/<filename>).
    uploaded_images = []
    for image_path in sorted(image_files):
        dest = f"{out_images_prefix.rstrip('/')}/{image_path.name}"
        bucket.blob(dest).upload_from_filename(str(image_path))
        uploaded_images.append(dest)

    # Bump docs_latest_version.json.
    version_blob = bucket.blob(version_blob_name)
    if version_blob.exists():
        data = json.loads(version_blob.download_as_text())
        if not isinstance(data, dict):
            data = {}
    else:
        data = {}
    current = int(data.get("doc_version", 0) or 0)
    new_version = current + 1
    data["doc_version"] = new_version
    version_blob.upload_from_string(
        json.dumps(data, indent=2) + "\n",
        content_type="application/json",
    )

    # Keep the local stamp in sync so the app does not re-download this upload.
    local_docs_dir = str(doc_path.parent)
    write_local_doc_version(new_version, docs_dir=local_docs_dir)

    print(f"Archived {len(archived)} existing object(s) under {archive_prefix}/{stamp}/")
    print(f"Uploaded {out_doc_blob}")
    print(f"Uploaded {len(uploaded_images)} image(s) under {out_images_prefix}/")
    print(f"Bumped {version_blob_name}: doc_version {current} -> {new_version}")
    print(f"Wrote local docs_version.json ({new_version}) under {local_docs_dir}")
    return new_version


def get_video_path(video_name):
    """
    Get the correct path for a video file.
    First, check if the video exists in the originally selected location.
    If not found, check in the data directory.
    """
    # First, check if the video_name is already a full path
    if os.path.isfile(video_name):
        return video_name
    return resource_path(os.path.join('data', video_name))
