"""
Ingestion: walk a repository and turn source files into retrieval-ready chunks.

Strategy
--------
1. Preferred path: parse the file with tree-sitter and cut along real AST
   boundaries (functions, classes, blocks). Sibling nodes are merged greedily
   up to `chunk_size` (bytes); oversized nodes are descended into recursively.
2. Fallback path (grammar unavailable / parse error / docs): LangChain's
   syntax-aware RecursiveCharacterTextSplitter, with start offsets mapped
   back to line numbers.

Every chunk carries strict provenance metadata:
    source / file_path  repo-relative POSIX path
    file_name           basename
    language            python | csharp | javascript | typescript | java |
                        go | rust | cpp | c | markdown | rst
    start_line/end_line 1-based, inclusive
    chunk_id            unique, stable id (used for Chroma ids and RRF fusion)

Install the grammars with:  pip install tree-sitter tree-sitter-language-pack
"""
from __future__ import annotations

import fnmatch
import hashlib
import os
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

try:  # optional: falls back to regex-aware splitting if missing
    from tree_sitter_language_pack import get_parser
except ImportError:  # pragma: no cover
    get_parser = None

CHUNKER_VERSION = "5"  # bump to invalidate cached indexes when chunking logic changes
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 150
MAX_FILE_BYTES = 1_000_000


@dataclass(frozen=True)
class LangSpec:
    name: str                       # value stored in metadata["language"]
    ts_grammar: Optional[str]       # tree-sitter grammar name (None = no AST path)
    splitter: Optional[Language]    # LangChain fallback splitter


LANGUAGE_CONFIG = {
    ".py": LangSpec("python", "python", Language.PYTHON),
    ".cs": LangSpec("csharp", "csharp", Language.CSHARP),
    ".js": LangSpec("javascript", "javascript", Language.JS),
    ".jsx": LangSpec("javascript", "javascript", Language.JS),
    ".mjs": LangSpec("javascript", "javascript", Language.JS),
    ".cjs": LangSpec("javascript", "javascript", Language.JS),
    ".ts": LangSpec("typescript", "typescript", Language.TS),
    ".tsx": LangSpec("typescript", "tsx", Language.TS),
    ".java": LangSpec("java", "java", Language.JAVA),
    ".go": LangSpec("go", "go", Language.GO),
    ".rs": LangSpec("rust", "rust", Language.RUST),
    ".cpp": LangSpec("cpp", "cpp", Language.CPP),
    ".cc": LangSpec("cpp", "cpp", Language.CPP),
    ".cxx": LangSpec("cpp", "cpp", Language.CPP),
    ".hpp": LangSpec("cpp", "cpp", Language.CPP),
    ".c": LangSpec("c", "c", Language.C),
    ".h": LangSpec("c", "c", Language.C),
    # Web / styling
    ".html": LangSpec("html", "html", Language.HTML),
    ".htm": LangSpec("html", "html", Language.HTML),
    ".css": LangSpec("css", "css", None),
    ".scss": LangSpec("scss", "scss", None),
    # Config / data
    ".json": LangSpec("json", "json", None),
    ".yaml": LangSpec("yaml", "yaml", None),
    ".yml": LangSpec("yaml", "yaml", None),
    ".sql": LangSpec("sql", "sql", None),
    # Documentation
    ".md": LangSpec("markdown", None, Language.MARKDOWN),
    ".markdown": LangSpec("markdown", None, Language.MARKDOWN),
    ".rst": LangSpec("rst", None, Language.RST),
}

SKIP_DIRS = {
    ".git", ".hg", ".svn", ".idea", ".vscode", ".venv", "venv", "env",
    "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", ".tox",
    "dist", "build", "target", "bin", "obj", "vendor", ".next", "chroma_db",
    ".repomind_repos",
}

# Generated / noisy files that would only pollute the index.
SKIP_FILES = {"package-lock.json", "composer.lock", "pnpm-lock.yaml", "yarn.lock"}


def is_skipped_name(name: str) -> bool:
    """True for lock files and minified assets that should never be indexed."""
    return name in SKIP_FILES or name.endswith((".min.js", ".min.css"))


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _line_starts(data) -> List[int]:
    """Offsets at which each line starts. Works for both str and bytes."""
    nl = b"\n" if isinstance(data, bytes) else "\n"
    starts, i = [0], data.find(nl)
    while i != -1:
        starts.append(i + 1)
        i = data.find(nl, i + 1)
    return starts


def _offset_to_line(starts: Sequence[int], offset: int) -> int:
    """1-based line number containing `offset`."""
    return bisect_right(starts, max(offset, 0))


def _matches_any(rel_path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(rel_path, p) or fnmatch.fnmatch(os.path.basename(rel_path), p) for p in patterns)


def iter_source_files(repo_path: str, exclude_patterns: Sequence[str] = ()) -> Iterator[Tuple[str, str]]:
    """
    Yield (absolute_path, repo_relative_posix_path) for every supported file, deterministically.
    `repo_path` may also be a single supported file, in which case only that file is yielded.
    """
    if os.path.isfile(repo_path):
        name = os.path.basename(repo_path)
        ext = os.path.splitext(name)[1].lower()
        if ext in LANGUAGE_CONFIG and not is_skipped_name(name):
            try:
                if os.path.getsize(repo_path) <= MAX_FILE_BYTES:
                    yield repo_path, name
            except OSError:
                pass
        return
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            ext = os.path.splitext(name)[1].lower()
            if ext not in LANGUAGE_CONFIG or is_skipped_name(name):
                continue
            abs_path = os.path.join(root, name)
            rel = os.path.relpath(abs_path, repo_path).replace(os.sep, "/")
            if _matches_any(rel, exclude_patterns):
                continue
            try:
                if os.path.getsize(abs_path) > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield abs_path, rel


def compute_repo_fingerprint(
    repo_path: str,
    exclude_patterns: Sequence[str] = (),
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> str:
    """
    Cheap content-change fingerprint (paths + sizes + mtimes + chunker config).
    Used as the cache key for the index so edits invalidate it automatically.
    The config part also covers overlap and whether tree-sitter is installed,
    because both change the produced chunks.
    """
    if not os.path.exists(repo_path):
        raise FileNotFoundError(f"Path '{repo_path}' does not exist.")
    h = hashlib.sha1()
    h.update(
        f"{CHUNKER_VERSION}|{chunk_size}|{chunk_overlap}|ts={get_parser is not None}"
        f"|{os.path.abspath(repo_path)}".encode()
    )
    for abs_path, rel in iter_source_files(repo_path, exclude_patterns):
        try:
            st = os.stat(abs_path)
        except OSError:
            continue
        h.update(f"{rel}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()


@lru_cache(maxsize=None)
def _get_parser(grammar: str):
    if get_parser is None:
        return None
    try:
        return get_parser(grammar)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
# AST chunking (tree-sitter)
# ──────────────────────────────────────────────────────────────
def _ast_units(node, max_bytes: int) -> Iterator[Tuple[int, int]]:
    """Yield byte spans of the largest AST nodes that fit in `max_bytes`."""
    if node.end_byte - node.start_byte <= max_bytes or node.child_count == 0:
        yield node.start_byte, node.end_byte
        return
    for child in node.children:
        yield from _ast_units(child, max_bytes)


def _merge_units(units: Sequence[Tuple[int, int]], max_bytes: int) -> List[Tuple[int, int]]:
    """Greedily merge consecutive spans while the merged span stays within `max_bytes`."""
    merged: List[Tuple[int, int]] = []
    cur_s = cur_e = None
    for s, e in units:
        if cur_s is None:
            cur_s, cur_e = s, e
        elif e - cur_s <= max_bytes:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    if cur_s is not None:
        merged.append((cur_s, cur_e))
    return merged


def _hard_split(data: bytes, start: int, end: int, max_bytes: int) -> Iterator[Tuple[int, int]]:
    """
    Last resort for a single huge leaf (e.g. minified line): cut on newlines,
    else by size. A size cut is moved back to a UTF-8 character boundary so
    multi-byte characters are never split (which produced U+FFFD garbage).
    """
    while end - start > max_bytes:
        limit = start + max_bytes
        cut = data.rfind(b"\n", start, limit)
        if cut <= start:
            cut = limit
            while cut > start and (data[cut] & 0xC0) == 0x80:  # continuation byte -> step back
                cut -= 1
            if cut <= start:  # pathological: no boundary found, fall back to raw cut
                cut = limit
        else:
            cut += 1
        yield start, cut
        start = cut
    if start < end:
        yield start, end


def _trim(data: bytes, start: int, end: int) -> Tuple[int, int]:
    while start < end and data[start:start + 1].isspace():
        start += 1
    while end > start and data[end - 1:end].isspace():
        end -= 1
    return start, end


def _extend_to_line_start(data: bytes, start: int) -> int:
    """
    If only spaces/tabs sit between `start` and the beginning of its line, move `start`
    back to the line start so the chunk keeps its indentation (important for Python).
    Line numbers are unaffected. Mid-line starts are left alone.
    """
    i = start
    while i > 0 and data[i - 1:i] in (b" ", b"\t"):
        i -= 1
    return i if (i == 0 or data[i - 1:i] == b"\n") else start


def _chunk_with_tree_sitter(source: str, spec: LangSpec, chunk_size: int) -> Optional[List[Tuple[str, int, int]]]:
    """Return [(text, start_line, end_line)] or None if the AST path is unavailable."""
    if not spec.ts_grammar:
        return None
    parser = _get_parser(spec.ts_grammar)
    if parser is None:
        return None
    try:
        data = source.encode("utf-8")
        tree = parser.parse(data)
        spans = _merge_units(list(_ast_units(tree.root_node, chunk_size)), chunk_size)
    except Exception:  # includes RecursionError on pathological nesting -> use fallback splitter
        return None

    starts = _line_starts(data)
    out: List[Tuple[str, int, int]] = []
    for s, e in spans:
        sub_spans = _hard_split(data, s, e, chunk_size) if e - s > chunk_size else [(s, e)]
        for ss, ee in sub_spans:
            ss, ee = _trim(data, ss, ee)
            if ss >= ee:
                continue
            ss = _extend_to_line_start(data, ss)
            text = data[ss:ee].decode("utf-8", errors="replace")
            out.append((text, _offset_to_line(starts, ss), _offset_to_line(starts, ee - 1)))
    return out


# ──────────────────────────────────────────────────────────────
# Fallback / documentation chunking
# ──────────────────────────────────────────────────────────────
def _chunk_with_splitter(source: str, spec: LangSpec, chunk_size: int, chunk_overlap: int) -> List[Tuple[str, int, int]]:
    kwargs = dict(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
    if spec.splitter is not None:
        splitter = RecursiveCharacterTextSplitter.from_language(spec.splitter, **kwargs)
    else:
        splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], **kwargs)

    starts = _line_starts(source)
    out = []
    for piece in splitter.create_documents([source]):
        start = max(piece.metadata.get("start_index", 0), 0)
        end = start + len(piece.page_content) - 1
        out.append((piece.page_content, _offset_to_line(starts, start), _offset_to_line(starts, end)))
    return out


def _make_doc(text: str, rel_path: str, spec: LangSpec, start_line: int, end_line: int, idx: int) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": rel_path,
            "file_path": rel_path,
            "file_name": os.path.basename(rel_path),
            "language": spec.name,
            "start_line": start_line,
            "end_line": end_line,
            "chunk_id": f"{rel_path}:{start_line}-{end_line}#{idx}",
        },
    )


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────
def load_and_chunk_repo(
    repo_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    exclude_patterns: Sequence[str] = (),
) -> List[Document]:
    """Ingest a repository into provenance-rich chunks (code + markdown docs)."""
    if not os.path.exists(repo_path):
        raise FileNotFoundError(f"Path '{repo_path}' does not exist.")

    docs: List[Document] = []
    for abs_path, rel in iter_source_files(repo_path, exclude_patterns):
        spec = LANGUAGE_CONFIG[os.path.splitext(abs_path)[1].lower()]
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                source = fh.read()
        except OSError:
            continue
        if not source.strip():
            continue

        try:
            pieces = _chunk_with_tree_sitter(source, spec, chunk_size)
            if pieces is None:
                pieces = _chunk_with_splitter(source, spec, chunk_size, chunk_overlap)
        except Exception as exc:  # one bad file must never break indexing
            print(f"[chunker] skipped {rel}: {exc}")
            continue

        for idx, (text, s_line, e_line) in enumerate(pieces):
            docs.append(_make_doc(text, rel, spec, s_line, e_line, idx))
    return docs


if __name__ == "__main__":
    from collections import Counter

    chunks = load_and_chunk_repo("./data")
    print(f"{len(chunks)} chunks | tree-sitter available: {get_parser is not None}")
    print(Counter(c.metadata["language"] for c in chunks))