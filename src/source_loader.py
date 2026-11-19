"""
Source loading: turn whatever the user typed or uploaded into a local path the chunker can index.

Accepted inputs
---------------
* a local folder                     -> used as is
* a single supported source file     -> indexed on its own
* a local .zip file                  -> safely extracted into the cache folder
* an uploaded .zip or source file    -> (Streamlit file_uploader) saved / extracted into the cache folder
* a public GitHub URL                -> shallow `git clone` into the cache folder
      https://github.com/owner/repo
      https://github.com/owner/repo.git
      https://github.com/owner/repo/tree/<branch>[/<sub/folder>]

Everything created by this module lives under REPOS_DIR (default ./.repomind_repos), in folders
whose names are derived from content hashes or repo names, so repeated use hits the index cache.

Safety notes
------------
* git is called with an argument list (never a shell string), and only for github.com URLs.
* Zip extraction rejects absolute paths and ".." (zip-slip), skips symlinks, extracts only
  supported source files, and enforces file-count / size limits (zip-bomb guard).
* GIT_TERMINAL_PROMPT=0 so a private repo fails fast instead of hanging on a password prompt.
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import stat
import subprocess
import zipfile
from dataclasses import dataclass, field
from typing import List, Optional

from src.chunker import LANGUAGE_CONFIG, MAX_FILE_BYTES, SKIP_DIRS, is_skipped_name

REPOS_DIR = os.environ.get("REPOMIND_REPOS_DIR", "./.repomind_repos")
MAX_ZIP_FILES = 20_000
MAX_ZIP_TOTAL_BYTES = 500 * 1024 * 1024
GIT_TIMEOUT_SECONDS = 300

_GH_URL = re.compile(
    r"^https?://(?:www\.)?github\.com/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?(?P<rest>/.*)?$",
    re.IGNORECASE,
)
_SAFE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class SourceError(ValueError):
    """A user-facing problem with the provided source (message is safe to show in the UI)."""


@dataclass
class ResolvedSource:
    path: str                 # folder or file to hand to the chunker
    label: str                # short human-readable description for the UI
    kind: str                 # folder | file | zip | github
    notes: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────
def supported_extensions() -> List[str]:
    return sorted(LANGUAGE_CONFIG)


def _is_supported_file(name: str) -> bool:
    ext = os.path.splitext(name)[1].lower()
    return ext in LANGUAGE_CONFIG and not is_skipped_name(name)


def _rmtree(path: str) -> None:
    """rmtree that also copes with read-only files (common inside .git on Windows)."""
    if not os.path.exists(path):
        return

    def _fix(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass

    shutil.rmtree(path, onerror=_fix)


def _hash_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.basename(name)) or "file"


# ──────────────────────────────────────────────────────────────
# Zip handling
# ──────────────────────────────────────────────────────────────
def _copy_limited(src, out, limit: int) -> int:
    """Copy a zip member while enforcing a real byte limit (headers can lie about size)."""
    written = 0
    while True:
        block = src.read(1 << 16)
        if not block:
            return written
        written += len(block)
        if written > limit:
            raise SourceError("A file inside the zip is larger than the allowed size limit.")
        out.write(block)


def _extract_zip(zf: zipfile.ZipFile, digest: str) -> str:
    """Safely extract supported source files; reuse an earlier extraction of identical content."""
    # The signature makes old extractions (made when fewer file types were supported) get redone.
    sig = hashlib.sha1(",".join(supported_extensions()).encode()).hexdigest()[:6]
    dest = os.path.join(REPOS_DIR, f"zip__{digest[:16]}_{sig}")
    marker = os.path.join(dest, ".repomind_complete")
    if os.path.exists(marker):
        return dest

    os.makedirs(REPOS_DIR, exist_ok=True)
    tmp = dest + ".tmp"
    _rmtree(tmp)
    _rmtree(dest)
    os.makedirs(tmp)
    root = os.path.realpath(tmp)

    count = 0
    total = 0
    try:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace("\\", "/")
            parts = [p for p in name.split("/") if p not in ("", ".")]
            if not parts:
                continue
            if name.startswith("/") or ".." in parts or re.match(r"^[A-Za-z]:", parts[0]):
                raise SourceError("The zip contains unsafe file paths and was rejected.")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:  # symlink
                continue
            if any(p in SKIP_DIRS or p == "__MACOSX" for p in parts[:-1]):
                continue
            if not _is_supported_file(parts[-1]) or info.file_size > MAX_FILE_BYTES:
                continue

            count += 1
            if count > MAX_ZIP_FILES:
                raise SourceError(f"The zip has more than {MAX_ZIP_FILES} source files.")

            target = os.path.realpath(os.path.join(root, *parts))
            if not target.startswith(root + os.sep):
                raise SourceError("The zip contains unsafe file paths and was rejected.")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                total += _copy_limited(src, out, MAX_FILE_BYTES)
            if total > MAX_ZIP_TOTAL_BYTES:
                raise SourceError("The zip's source files are too large to index.")

        if count == 0:
            raise SourceError(
                "The zip contains no supported source files "
                f"(supported: {', '.join(supported_extensions())})."
            )
        with open(os.path.join(tmp, ".repomind_complete"), "w", encoding="utf-8") as fh:
            fh.write("ok")
        os.replace(tmp, dest)
    except Exception:
        _rmtree(tmp)
        raise
    return dest


def _from_zip_path(path: str) -> ResolvedSource:
    try:
        digest = _hash_file(path)
        with zipfile.ZipFile(path) as zf:
            dest = _extract_zip(zf, digest)
    except zipfile.BadZipFile:
        raise SourceError(f"'{path}' is not a valid zip file.")
    return ResolvedSource(dest, f"zip: {os.path.basename(path)}", "zip")


def _from_upload(name: str, data: bytes) -> ResolvedSource:
    ext = os.path.splitext(name)[1].lower()
    digest = hashlib.sha1(data).hexdigest()

    if ext == ".zip":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                dest = _extract_zip(zf, digest)
        except zipfile.BadZipFile:
            raise SourceError("The uploaded file is not a valid zip.")
        return ResolvedSource(dest, f"upload: {_safe_name(name)}", "zip")

    if _is_supported_file(name):
        folder = os.path.join(REPOS_DIR, f"upload__{digest[:16]}")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, _safe_name(name))
        with open(path, "wb") as fh:
            fh.write(data)
        return ResolvedSource(path, f"file: {_safe_name(name)}", "file")

    raise SourceError(
        f"'{name}' isn't a supported upload. Use a .zip or one of: {', '.join(supported_extensions())}."
    )


# ──────────────────────────────────────────────────────────────
# GitHub handling
# ──────────────────────────────────────────────────────────────
def _git(args: List[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise SourceError("git took too long (timed out). Try a smaller repo or a sub-folder URL.")


def _git_error_message(stderr: str) -> str:
    text = (stderr or "").strip()
    low = text.lower()
    if "not found" in low and "branch" in low:
        return "That branch wasn't found. (Branch names containing '/' aren't supported in /tree/ URLs.)"
    if "not found" in low or "could not read" in low or "authentication" in low or "terminal prompts disabled" in low:
        return "Repository not found, or it is private. Only public GitHub repositories are supported."
    last = text.splitlines()[-1] if text else "unknown error"
    return f"git failed: {last}"


def _from_github(url: str) -> ResolvedSource:
    clean = url.split("#", 1)[0].split("?", 1)[0].rstrip("/")
    m = _GH_URL.match(clean)
    if not m:
        raise SourceError("Only public GitHub URLs are supported, e.g. https://github.com/owner/repo")

    owner, repo = m.group("owner"), m.group("repo")
    if owner in (".", "..") or repo in (".", ".."):
        raise SourceError("That GitHub URL looks invalid.")

    branch: Optional[str] = None
    sub = ""
    parts = [p for p in (m.group("rest") or "").split("/") if p]
    if parts:
        if parts[0] in ("tree", "blob") and len(parts) >= 2:
            branch = parts[1]
            sub_parts = parts[2:]
            if not _SAFE_REF.match(branch) or ".." in sub_parts:
                raise SourceError("That GitHub URL looks invalid.")
            sub = "/".join(sub_parts)
        else:
            raise SourceError(
                "Unsupported GitHub URL. Use https://github.com/owner/repo or .../tree/<branch>/<folder>."
            )

    if shutil.which("git") is None:
        raise SourceError("Git isn't installed (or not on PATH), so GitHub URLs can't be cloned.")

    os.makedirs(REPOS_DIR, exist_ok=True)
    dest = os.path.join(REPOS_DIR, f"github__{owner}__{repo}" + (f"__{branch}" if branch else ""))
    notes: List[str] = []

    if os.path.isdir(os.path.join(dest, ".git")):
        fetched = _git(["fetch", "--depth", "1", "origin"], cwd=dest)
        reset = _git(["reset", "--hard", "FETCH_HEAD"], cwd=dest) if fetched.returncode == 0 else fetched
        if reset.returncode != 0:
            notes.append("Couldn't update the cached clone from GitHub; using the previously downloaded copy.")
    else:
        tmp = dest + ".tmp"
        _rmtree(tmp)
        args = ["clone", "--depth", "1", "--single-branch"]
        if branch:
            args += ["--branch", branch]
        args += ["--", f"https://github.com/{owner}/{repo}.git", tmp]
        proc = _git(args)
        if proc.returncode != 0:
            _rmtree(tmp)
            raise SourceError(_git_error_message(proc.stderr))
        _rmtree(dest)
        os.replace(tmp, dest)

    target = dest
    if sub:
        base = os.path.realpath(dest)
        target = os.path.realpath(os.path.join(dest, *sub.split("/")))
        if not (target == base or target.startswith(base + os.sep)) or not os.path.exists(target):
            raise SourceError(f"The path '{sub}' doesn't exist in that repository.")

    label = f"GitHub: {owner}/{repo}" + (f"@{branch}" if branch else "") + (f"/{sub}" if sub else "")
    return ResolvedSource(target, label, "github", notes)


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────
def resolve_source(text: str = "", uploaded=None) -> ResolvedSource:
    """
    Resolve user input into a local path for indexing.

    `uploaded` may be a Streamlit UploadedFile (anything with `.name` and `.getvalue()`);
    when given, it takes priority over `text`.
    Raises SourceError with a readable message when the input can't be used.
    """
    if uploaded is not None:
        return _from_upload(uploaded.name, uploaded.getvalue())

    raw = (text or "").strip().strip('"').strip("'").strip()
    if not raw:
        raise SourceError("Enter a folder, file, .zip path or GitHub URL, or upload a file.")

    if re.match(r"^https?://", raw, re.IGNORECASE):
        return _from_github(raw)
    if raw.lower().startswith(("git@", "ssh://")):
        raise SourceError("SSH URLs aren't supported. Use the https:// URL of a public repository.")

    path = os.path.expanduser(raw)
    if os.path.isdir(path):
        return ResolvedSource(path, os.path.basename(os.path.abspath(path)) or path, "folder")

    if os.path.isfile(path):
        if path.lower().endswith(".zip"):
            return _from_zip_path(path)
        if _is_supported_file(path):
            return ResolvedSource(path, f"file: {os.path.basename(path)}", "file")
        raise SourceError(
            f"'{os.path.basename(path)}' isn't a supported source file. "
            f"Supported: {', '.join(supported_extensions())} (or a .zip)."
        )

    raise SourceError(f"'{raw}' is not a folder, file, .zip or GitHub URL that exists on this machine.")