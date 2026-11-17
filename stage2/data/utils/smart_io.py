"""Transparent local/S3/BOS path handling for Motus2 data loading.

Two remote loading strategies (controlled by cache_dir in smart_load_pt):

1. Streaming (cache_dir=None):
     megfile.smart_open(path, "rb") → io.BytesIO → torch.load
   No local disk written. Requires `pip install megfile` with BOS credentials configured.

2. Download-and-cache (cache_dir set):
     bcecmd bos cp (preferred) or megfile cp → local file → torch.load
   File downloaded once, reused on subsequent accesses.
"""

import io
import json
import errno
import os
import re
import shutil
import subprocess
import logging
import time
from pathlib import Path
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path predicates
# ---------------------------------------------------------------------------

def _normalize_bos(path: str) -> str:
    """Convert bos:// or bos:/ to s3:// for megfile compatibility.

    megfile only recognises the s3:// scheme; bos:// causes
    'protocol bos not found'. This mirrors realworld_dataset._normalize_bos.
    """
    if path.startswith("bos://"):
        return "s3://" + path[len("bos://"):]
    if path.startswith("bos:/"):
        return "s3://" + path[len("bos:/"):]
    return path


def is_remote(path: str) -> bool:
    """Return True for s3:// or bos:/ paths."""
    return isinstance(path, str) and (
        path.startswith("s3://") or path.startswith("bos://") or path.startswith("bos:/")
    )


def smart_join(*parts: str) -> str:
    """Join path components for both local and remote paths."""
    if not parts:
        return ""
    base = parts[0]
    if is_remote(base):
        # Preserve the leading scheme (e.g. "s3://", "bos://") from the first part,
        # then strip slashes only from the remaining parts.
        scheme_end = base.index("://") + 3  # position after "://"
        prefix = base[:scheme_end]          # e.g. "s3://"
        rest = [base[scheme_end:].rstrip("/")] + [p.strip("/") for p in parts[1:] if p]
        return prefix + "/".join(rest)
    return str(Path(*parts))


def stem(path: str) -> str:
    """Return filename stem (no directory, no extension)."""
    return Path(path).stem


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

def _bcecmd_available() -> bool:
    return shutil.which("bcecmd") is not None


def _megfile_available() -> bool:
    try:
        import megfile  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_file(remote: str, local: str) -> None:
    """Download a single remote file to a local path.

    Uses bcecmd if available (preferred for Baidu BOS); otherwise megfile CLI.
    Raises RuntimeError on failure.
    """
    os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)

    if _bcecmd_available():
        bos_url = remote.replace("s3://", "bos:/")
        cmds = [f"bcecmd bos cp {bos_url} {local} --yes --disable-bar --quiet"]
        if shutil.which("megfile"):
            cmds.append(f"megfile cp {remote!r} {local!r}")
    elif shutil.which("megfile"):
        cmds = [f"megfile cp {remote!r} {local!r}"]
    else:
        raise RuntimeError(
            f"Cannot download {remote}: neither bcecmd nor megfile CLI found. "
            "Install megfile (`pip install megfile`) or bcecmd."
        )

    max_retries = 6
    last_error = None
    for attempt in range(max_retries):
        for cmd in cmds:
            ret = subprocess.run(cmd, shell=True, capture_output=True)
            if ret.returncode == 0:
                return
            last_error = (
                f"Download failed (exit {ret.returncode}): {cmd}\n"
                f"stderr: {ret.stderr.decode('utf-8', errors='replace')}"
            )
        if attempt < max_retries - 1:
            wait = 0.5 * (2 ** attempt)
            logger.warning(
                f"Remote download retry {attempt + 1}/{max_retries} for {remote} in {wait:.1f}s"
            )
            time.sleep(wait)
    raise RuntimeError(last_error or f"Download failed: {remote}")


def _remote_to_local(remote: str, cache_dir: str) -> str:
    """Map a remote URL to its local cache path.

    s3://bucket/a/b/c.pt + cache_dir  →  cache_dir/bucket/a/b/c.pt
    bos:/bucket/a/b/c.pt  + cache_dir  →  cache_dir/bucket/a/b/c.pt
    """
    # Strip scheme
    rel = re.sub(r'^(s3://|bos://|bos:/)', '', remote).lstrip("/")
    return os.path.join(cache_dir, rel)


def _stream_fallback_cache_dir() -> str:
    return os.environ.get(
        "MOTUS2_STREAM_FALLBACK_CACHE_DIR",
        "/tmp/motus2_stream_fallback",
    )


def _stream_timeout_sec(default: int = 60) -> int:
    raw = os.environ.get("MOTUS2_STREAM_TIMEOUT_SEC")
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid MOTUS2_STREAM_TIMEOUT_SEC=%r, fallback to %ss", raw, default)
        return default
    if value <= 0:
        logger.warning("Non-positive MOTUS2_STREAM_TIMEOUT_SEC=%r, fallback to %ss", raw, default)
        return default
    return value


def _stream_max_retries(default: int = 1) -> int:
    raw = os.environ.get("MOTUS2_STREAM_MAX_RETRIES")
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid MOTUS2_STREAM_MAX_RETRIES=%r, fallback to %s", raw, default)
        return default
    if value <= 0:
        logger.warning("Non-positive MOTUS2_STREAM_MAX_RETRIES=%r, fallback to %s", raw, default)
        return default
    return value


def _is_retryable_stream_error(error: Exception) -> bool:
    text = repr(error)
    retry_markers = (
        "EndpointConnectionError",
        "ConnectTimeoutError",
        "ReadTimeoutError",
        "ConnectionClosedError",
        "Connection reset",
        "Connection aborted",
        "temporarily unavailable",
        "timed out",
        "Could not connect",
        "Broken pipe",
        "IncompleteRead",
        "RemoteDisconnected",
    )
    return any(marker in text for marker in retry_markers)


def _is_missing_remote_error(error: Exception) -> bool:
    text = repr(error)
    missing_markers = (
        "No such file",
        "NoSuchKey",
        "404",
        "Not Found",
        "404 Not Found",
        "FileNotFoundError",
    )
    return any(marker in text for marker in missing_markers)


def _is_corrupt_pt_error(error: Exception) -> bool:
    text = repr(error)
    markers = (
        "PytorchStreamReader failed reading zip archive",
        "failed finding central directory",
        "invalid header",
        "not a ZIP archive",
        "filename 'storages' not found",
        "failed locating file",
    )
    return any(marker in text for marker in markers)


def _load_via_fallback_cache(path: str, map_location="cpu"):
    cache_dir = _stream_fallback_cache_dir()
    local = _remote_to_local(path, cache_dir)
    if not os.path.exists(local):
        logger.warning(f"Falling back to cached download for remote PT: {path}")
        _download_file(path, local)
    return _torch_load_data(local, map_location=map_location)


# ---------------------------------------------------------------------------
# Smart load
# ---------------------------------------------------------------------------

def _torch_load_data(path_or_buffer, map_location="cpu"):
    """Load arbitrary dataset .pt payloads across PyTorch versions."""
    try:
        return torch.load(path_or_buffer, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path_or_buffer, map_location=map_location)


def _stream_load_pt(
    path: str,
    map_location="cpu",
    max_retries: int = 2,
    stream_timeout: int = 60,
    allow_fallback_cache: bool = True,
):
    """Stream a .pt file directly from BOS/S3 via megfile.smart_open.

    No local file is written; data is read into memory via io.BytesIO and
    passed to torch.load.
    Requires megfile with BOS credentials configured.
    Retries with exponential backoff on 403/throttling/timeout errors.
    """
    import signal
    try:
        from megfile import smart_open
    except ImportError:
        raise RuntimeError(
            f"megfile is required for streaming BOS paths (cache_dir=None). "
            "Install with: pip install megfile"
        )

    class _StreamTimeout(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise _StreamTimeout(f"BOS streaming timed out after {stream_timeout}s: {path}")

    for attempt in range(max_retries):
        try:
            # Set alarm to detect hanging BOS connections (only works in main thread)
            old_handler = None
            try:
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(stream_timeout)
            except (ValueError, OSError):
                pass  # Not in main thread (DataLoader worker) — skip alarm

            with smart_open(path, "rb") as f:
                data = _torch_load_data(io.BytesIO(f.read()), map_location=map_location)

            # Cancel alarm
            try:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
            except (ValueError, OSError):
                pass
            return data
        except _StreamTimeout:
            if attempt < max_retries - 1:
                logger.warning(
                    f"BOS stream timeout on {path}, retrying {attempt+1}/{max_retries - 1}"
                )
                wait = 0.5 * (2 ** attempt)
                time.sleep(wait)
                continue
            logger.warning(f"BOS stream timeout on {path}, no retries left")
            if allow_fallback_cache:
                return _load_via_fallback_cache(path, map_location=map_location)
            raise
        except Exception as e:
            # Cancel alarm on error
            try:
                signal.alarm(0)
            except (ValueError, OSError):
                pass
            if _is_missing_remote_error(e):
                raise
            if attempt < max_retries - 1 and (
                "403" in str(e)
                or "Forbidden" in str(e)
                or "throttl" in str(e).lower()
                or _is_retryable_stream_error(e)
            ):
                wait = 0.5 * (2 ** attempt)
                logger.warning(
                    f"BOS stream retrying {attempt+1}/{max_retries - 1} on {path} in {wait:.1f}s: {e}"
                )
                time.sleep(wait)
            else:
                if _is_retryable_stream_error(e) and allow_fallback_cache:
                    logger.warning(
                        f"BOS stream failed on final attempt for {path}, falling back to cache: {e}"
                    )
                    return _load_via_fallback_cache(path, map_location=map_location)
                raise

    if allow_fallback_cache:
        return _load_via_fallback_cache(path, map_location=map_location)
    raise RuntimeError(f"BOS streaming failed after {max_retries} retries: {path}")


def smart_load_json(path: str) -> dict:
    """Load a JSON file from a local or remote (S3/BOS) path.

    Remote paths are streamed into memory via megfile.smart_open.
    Used to download manifest.json with a single BOS API call.
    """
    if not is_remote(path):
        with open(path, "r") as f:
            return json.load(f)
    path = _normalize_bos(path)
    try:
        from megfile import smart_open
    except ImportError:
        raise RuntimeError(
            "megfile is required to load remote JSON (manifest). "
            "Install with: pip install megfile"
        )
    with smart_open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def smart_load_pt(path: str, cache_dir: Optional[str] = None, map_location="cpu"):
    """Load a .pt file from a local or remote (S3/BOS) path.

    Two remote strategies (controlled by cache_dir):
    - cache_dir=None  → stream directly via megfile.smart_open, no local disk written.
    - cache_dir set   → download to local cache on first access (bcecmd preferred),
                        reuse cached file on subsequent accesses.

    Args:
        path: Local filesystem path or s3:// / bos:// URL.
        cache_dir: Root dir for downloaded files. None means streaming mode.
        map_location: Passed to torch.load.
    """
    if not is_remote(path):
        return _torch_load_data(path, map_location=map_location)

    path = _normalize_bos(path)

    if cache_dir is None:
        return _stream_load_pt(
            path,
            map_location,
            max_retries=_stream_max_retries(),
            stream_timeout=_stream_timeout_sec(),
            allow_fallback_cache=False,
        )

    local = _remote_to_local(path, cache_dir)
    if not os.path.exists(local):
        logger.debug(f"Downloading {path} → {local}")
        _download_file(path, local)

    try:
        return _torch_load_data(local, map_location=map_location)
    except Exception as e:
        # Cache files on shared FS can occasionally be truncated/corrupted by
        # interrupted writes. If that happens, drop the bad cache and redownload once.
        is_errno_22 = isinstance(e, OSError) and getattr(e, "errno", None) == errno.EINVAL
        if _is_corrupt_pt_error(e) or is_errno_22:
            logger.warning("Corrupted cached PT detected, redownloading: %s (%s)", local, e)
            try:
                os.remove(local)
            except OSError:
                pass
            _download_file(path, local)
            return _torch_load_data(local, map_location=map_location)
        raise


# ---------------------------------------------------------------------------
# Smart directory operations
# ---------------------------------------------------------------------------

def smart_isdir(path: str) -> bool:
    """Check whether path is an existing directory (local or remote)."""
    if not is_remote(path):
        return os.path.isdir(path)
    path = _normalize_bos(path)
    if _megfile_available():
        from megfile import smart_isdir as _isdir
        return _isdir(path)
    # Fall back: try listing; non-empty → dir exists
    return len(_bcecmd_ls(path)) > 0


def smart_listdir(path: str) -> List[str]:
    """Return list of immediate children of a directory as full paths.

    Returns [] if the directory does not exist or cannot be listed.
    """
    if not is_remote(path):
        p = Path(path)
        if not p.is_dir():
            return []
        return [str(c) for c in sorted(p.iterdir())]

    path = _normalize_bos(path)
    if _megfile_available():
        try:
            from megfile import smart_listdir as _listdir
            prefix = path.rstrip("/")
            return [f"{prefix}/{name}" for name in sorted(_listdir(path))]
        except Exception as e:
            logger.warning(f"megfile.smart_listdir({path}) failed: {e}")

    return _bcecmd_ls(path)


def smart_listdir_dirs(path: str) -> List[str]:
    """Return only the subdirectories under path."""
    children = smart_listdir(path)
    if not is_remote(path):
        return [c for c in children if os.path.isdir(c)]
    # Remote: megfile / bcecmd listings include trailing '/' for dirs
    if _megfile_available():
        try:
            from megfile import smart_isdir as _isdir
            return [c for c in children if _isdir(c)]
        except Exception:
            pass
    # Heuristic: BOS ls output has trailing '/' for prefixes (pseudo-dirs)
    return [c for c in children if c.endswith("/") or "/" not in Path(c).suffix]


def smart_glob_pt(dir_path: str) -> List[str]:
    """Return all *.pt files directly under dir_path."""
    if not is_remote(dir_path):
        p = Path(dir_path)
        if not p.is_dir():
            return []
        return sorted(str(f) for f in p.glob("*.pt"))

    dir_path = _normalize_bos(dir_path)
    if _megfile_available():
        try:
            from megfile import smart_glob
            return sorted(smart_glob(f"{dir_path.rstrip('/')}/*.pt"))
        except Exception as e:
            logger.warning(f"megfile.smart_glob({dir_path}) failed: {e}")

    # Fall back: list dir and filter
    return sorted(c for c in smart_listdir(dir_path) if c.endswith(".pt"))


def smart_exists(path: str) -> bool:
    """Check whether a file or directory exists (local or remote)."""
    if not is_remote(path):
        return os.path.exists(path)
    path = _normalize_bos(path)
    if _megfile_available():
        from megfile import smart_exists as _exists
        return _exists(path)
    # Heuristic: listing the parent and checking membership
    parent = path.rstrip("/").rsplit("/", 1)[0]
    name = path.rstrip("/").rsplit("/", 1)[-1]
    return any(c.rstrip("/").endswith(name) for c in smart_listdir(parent))


# ---------------------------------------------------------------------------
# bcecmd listing helper
# ---------------------------------------------------------------------------

def _bcecmd_ls(path: str) -> List[str]:
    """List a BOS path using the bcecmd CLI. Returns full paths."""
    if not _bcecmd_available():
        return []
    bos_url = path.replace("s3://", "bos:/").rstrip("/") + "/"
    ret = subprocess.run(
        f"bcecmd bos ls {bos_url}",
        shell=True, capture_output=True, text=True
    )
    if ret.returncode != 0:
        return []
    lines = [l.strip() for l in ret.stdout.splitlines() if l.strip()]
    # bcecmd ls output: "bos:/bucket/prefix/name" or just the name depending on version
    results = []
    prefix = path.rstrip("/")
    for line in lines:
        # Strip leading metadata columns if present (date, size, key)
        parts = line.split()
        entry = parts[-1] if parts else ""
        if entry.startswith("bos:/") or entry.startswith("s3://"):
            results.append(entry.replace("bos:/", "s3://", 1))
        elif entry:
            results.append(f"{prefix}/{entry}")
    return sorted(results)
