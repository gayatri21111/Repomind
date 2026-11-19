import html
import inspect
import json
import os
import time
from collections import Counter

import streamlit as st
import streamlit.components.v1 as components

from src.chunker import LANGUAGE_CONFIG, compute_repo_fingerprint, load_and_chunk_repo
from src.rag_engine import (
    DEFAULT_MODEL,
    MIN_RERANK_SCORE,
    NOT_FOUND_MESSAGE,  # noqa: F401  (kept for convenience / external imports)
    ollama_health,
    stream_code_explanation,
)
from src.source_loader import SourceError, resolve_source
from src.vector_store import RETRIEVAL_MODES, build_hybrid_retriever

st.set_page_config(
    page_title="RepoMind — Ask your codebase anything",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXCLUDE_PATTERNS = ()  # e.g. ("tests/*", "*.generated.cs")
SAMPLE_QUESTIONS = [
    "Give me an overview of how this project is structured",
    "Where is the application entry point?",
    "How is error handling done?",
]
PRISM_LANG = {"csharp": "csharp", "cpp": "cpp", "rst": None}  # metadata language -> st.code language


# ──────────────────────────────────────────────────────────────
# Streamlit version compatibility
# `use_container_width` is deprecated/removed in newer Streamlit in favour of
# `width="stretch"`; older versions only know `use_container_width`.
# ──────────────────────────────────────────────────────────────
def _stretch_kwargs() -> dict:
    try:
        if "width" in inspect.signature(st.button).parameters:
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}


STRETCH = _stretch_kwargs()


# ──────────────────────────────────────────────────────────────
# Theme (light / dark) — Streamlit's own theme decides; we follow it
# ──────────────────────────────────────────────────────────────
def detect_theme():
    """'light' | 'dark' | None (unknown -> follow the OS via prefers-color-scheme)."""
    try:
        t = st.context.theme.type  # Streamlit >= 1.46
        return t if t in ("light", "dark") else None
    except Exception:
        return None


THEME = detect_theme()

LIGHT_TOKENS = """
--brand:#0b6e52; --brand-dark:#08543e; --brand-soft:#e7f4ee; --brand-line:#cfe8dc;
--ink:#14201b; --muted:#5f6f67; --line:#e1e8e4; --bg:#f5f8f6; --card:#ffffff;
--field:#ffffff; --field-line:#d3ddd8; --code-bg:#f3f7f5; --chip-bg:#f1f5f3;
--glow:#dff1e8; --shadow:rgba(20,32,27,.06); --answer:#26332d; --placeholder:#93a39b;
"""
DARK_TOKENS = """
--brand:#3ccf9f; --brand-dark:#8be8c9; --brand-soft:#12332a; --brand-line:#1f4d3f;
--ink:#e8f0ec; --muted:#9bb0a6; --line:#2a3631; --bg:#0e1512; --card:#151e1a;
--field:#101814; --field-line:#33433c; --code-bg:#0f1713; --chip-bg:#1a2420;
--glow:#12332a; --shadow:rgba(0,0,0,.45); --answer:#d5e2dc; --placeholder:#6f8379;
"""

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
/*TOKENS*/
html, body, .stApp, [class*="css"] { font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }
.stApp { background: radial-gradient(900px 380px at 85% -10%, var(--glow) 0%, transparent 70%), var(--bg); color: var(--ink); }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 880px; padding-top: 1.2rem; padding-bottom: 4rem; }
[data-testid="stSidebar"] { background: var(--card); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] * { color: var(--ink); }

.nav { display:flex; align-items:center; justify-content:space-between; padding:10px 0 18px; border-bottom:1px solid var(--line); margin-bottom:28px; }
.brand { display:flex; align-items:center; gap:10px; font-weight:800; font-size:1.25rem; color:var(--ink); }
.brand-mark { width:34px; height:34px; border-radius:10px; display:grid; place-items:center; background:linear-gradient(135deg,#12a47a,#0b6e52); color:#fff; font-size:1.05rem; box-shadow:0 4px 12px rgba(11,110,82,.35); }
.nav-badge { font-size:.78rem; font-weight:600; color:var(--brand); background:var(--brand-soft); padding:6px 12px; border-radius:999px; border:1px solid var(--brand-line); }

.stepper { display:flex; align-items:center; justify-content:center; margin:4px 0 30px; }
.step { display:flex; align-items:center; gap:10px; }
.dot { width:32px; height:32px; border-radius:50%; display:grid; place-items:center; font-weight:700; font-size:.85rem; border:2px solid var(--line); color:var(--muted); background:var(--card); }
.step.active .dot { border-color:#0b6e52; background:#0b6e52; color:#fff; box-shadow:0 0 0 5px rgba(11,110,82,.2); }
.step.done .dot { border-color:var(--brand); background:var(--brand-soft); color:var(--brand); }
.step-label { font-size:.88rem; font-weight:600; color:var(--muted); }
.step.active .step-label, .step.done .step-label { color:var(--ink); }
.bar { width:64px; height:2px; background:var(--line); margin:0 14px; border-radius:2px; }
.bar.done { background:var(--brand); }

.hero { text-align:center; margin-bottom:26px; }
.eyebrow { font-size:.78rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:var(--brand); }
.hero h1 { font-size:2.4rem; font-weight:800; letter-spacing:-.02em; margin:8px 0 10px; color:var(--ink); line-height:1.15; }
.hero p { color:var(--muted); font-size:1.05rem; max-width:560px; margin:0 auto; }

[data-testid="stVerticalBlockBorderWrapper"] { background:var(--card) !important; border:1px solid var(--line) !important; border-radius:20px !important; box-shadow:0 10px 30px var(--shadow); }
[data-testid="stForm"] { border:none !important; padding:0 !important; }
.card-title { font-size:1.15rem; font-weight:700; color:var(--ink); margin-bottom:2px; }
.card-sub { color:var(--muted); font-size:.92rem; margin-bottom:14px; }

.stTextInput label p { font-weight:600; font-size:.85rem; color:var(--ink); }
.stTextInput input { background:var(--field) !important; color:var(--ink) !important; border:1.5px solid var(--field-line) !important; border-radius:12px !important; padding:12px 14px !important; font-size:.98rem !important; }
.stTextInput input::placeholder { color:var(--placeholder) !important; }
.stTextInput input:focus { border-color:var(--brand) !important; box-shadow:0 0 0 4px rgba(11,110,82,.2) !important; }
.stTextInput [data-baseweb="input"], .stTextInput [data-baseweb="base-input"] { background:transparent !important; border:none !important; }

:is(.stButton,.stFormSubmitButton) > button { border-radius:12px; font-weight:600; padding:10px 20px; transition:all .15s ease; }
:is(.stButton,.stFormSubmitButton) > button:is([kind="primary"],[kind="primaryFormSubmit"],[data-testid="stBaseButton-primary"],[data-testid="stBaseButton-primaryFormSubmit"]) { background:linear-gradient(135deg,#12a47a,#0b6e52); color:#fff; border:none; box-shadow:0 6px 16px rgba(11,110,82,.28); }
:is(.stButton,.stFormSubmitButton) > button:is([kind="primary"],[kind="primaryFormSubmit"],[data-testid="stBaseButton-primary"],[data-testid="stBaseButton-primaryFormSubmit"]):hover:not(:disabled) { transform:translateY(-1px); box-shadow:0 10px 22px rgba(11,110,82,.34); color:#fff; }
:is(.stButton,.stFormSubmitButton) > button:is([kind="primary"],[kind="primaryFormSubmit"],[data-testid="stBaseButton-primary"],[data-testid="stBaseButton-primaryFormSubmit"]) p { color:#fff !important; }
:is(.stButton,.stFormSubmitButton) > button:is([kind="secondary"],[data-testid="stBaseButton-secondary"]) { background:var(--card); color:var(--brand); border:1.5px solid var(--brand-line); }
:is(.stButton,.stFormSubmitButton) > button:is([kind="secondary"],[data-testid="stBaseButton-secondary"]):hover { background:var(--brand-soft); border-color:var(--brand); color:var(--brand-dark); }
:is(.stButton,.stFormSubmitButton) > button:is([kind="secondary"],[data-testid="stBaseButton-secondary"]) p { color:inherit !important; font-weight:600; }

.stProgress > div > div > div > div { background:linear-gradient(90deg,#12a47a,#0b6e52); }
.chips { display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 18px; }
.chip { font-size:.78rem; font-weight:600; color:var(--muted); background:var(--chip-bg); border:1px solid var(--line); padding:4px 10px; border-radius:999px; display:inline-block; }
.chip.brand { color:var(--brand); background:var(--brand-soft); border-color:var(--brand-line); }
.loc { font-family:'JetBrains Mono',monospace; font-size:.8rem; color:var(--brand-dark); background:var(--brand-soft); border:1px solid var(--brand-line); border-radius:6px; padding:2px 8px; }

.section-head { display:flex; align-items:center; gap:8px; font-size:1.05rem; font-weight:700; margin:8px 0 10px; color:var(--ink); }
[data-testid="stChatMessage"] { background:transparent; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p, [data-testid="stChatMessage"] li { line-height:1.7; color:var(--answer); }
[data-testid="stMarkdownContainer"] p code, [data-testid="stMarkdownContainer"] li code { background:var(--brand-soft); color:var(--brand-dark); border-radius:6px; padding:2px 6px; font-family:'JetBrains Mono',monospace; font-size:.85em; border:1px solid var(--brand-line); }
[data-testid="stCode"], [data-testid="stCode"] pre { background:var(--code-bg) !important; border-radius:12px !important; border:1px solid var(--line); }
[data-testid="stCode"] code { font-family:'JetBrains Mono',monospace !important; font-size:.83rem !important; }
[data-testid="stExpander"] { border:1px solid var(--line) !important; border-radius:14px !important; background:var(--card); overflow:hidden; }
[data-testid="stExpander"] summary { font-weight:600; }
.hint { color:var(--muted); font-size:.82rem; text-align:center; margin-top:18px; }
hr { border-color:var(--line); }
"""


def build_css(theme):
    tokens = f":root{{{LIGHT_TOKENS}}}"
    if theme == "dark":
        tokens += f":root{{{DARK_TOKENS}}}"
    elif theme is None:  # unknown -> follow OS preference
        tokens += f"@media (prefers-color-scheme: dark){{:root{{{DARK_TOKENS}}}}}"
    return BASE_CSS.replace("/*TOKENS*/", tokens)


st.markdown(f"<style>{build_css(THEME)}</style>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Cached resources
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False, max_entries=3)
def build_index(repo_path: str, fingerprint: str):
    """
    Chunk + embed + BM25-index a repo once per (path, fingerprint).
    `fingerprint` changes whenever files change, which busts the cache; a plain
    page refresh or a new browser session reuses the in-memory index.
    Exceptions are not cached, so failed attempts can be retried.
    """
    chunks = load_and_chunk_repo(repo_path, exclude_patterns=EXCLUDE_PATTERNS)
    if not chunks:
        raise ValueError("No supported source files found in the specified path.")
    retriever = build_hybrid_retriever(chunks, fingerprint=fingerprint)
    retriever.warmup()
    stats = {
        "chunks": len(chunks),
        "files": len({c.metadata["file_path"] for c in chunks}),
        "languages": dict(Counter(c.metadata["language"] for c in chunks)),
    }
    return retriever, stats


@st.cache_data(ttl=10, show_spinner=False)
def cached_ollama_status():
    return ollama_health()


# ──────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────
for key, default in {
    "step": 1,
    "repo_path": "./data",  # resolved local folder/file that gets indexed
    "source_input": "./data",  # what the user typed (folder, file, zip path or GitHub URL)
    "source_label": "",
    "source_notes": [],
    "retriever": None,
    "index_stats": None,
    "messages": [],  # multi-turn memory: [{role, content, sources?, grounded?}]
}.items():
    st.session_state.setdefault(key, default)


def ask_sample(text: str):
    """Sample buttons actually ask the question (they used to only fill the text box)."""
    st.session_state.pending_q = text


def reset_to_start():
    st.session_state.step = 1
    st.session_state.messages = []
    st.session_state.retriever = None
    st.session_state.index_stats = None


# ──────────────────────────────────────────────────────────────
# UI components
# ──────────────────────────────────────────────────────────────
def copy_button(text: str, height: int = 40):
    """Clipboard button (works inside Streamlit's iframe sandbox via execCommand fallback)."""
    payload = json.dumps(text).replace("</", "<\\/")
    ink = "#e8f0ec" if THEME == "dark" else "#14201b"
    dark_media = (
        "@media (prefers-color-scheme: dark){button{color:#e8f0ec;border-color:#33433c}}" if THEME is None else ""
    )
    components.html(
        f"""
<style>
  body{{margin:0;font-family:Inter,system-ui,sans-serif}}
  button{{background:transparent;color:{ink};border:1px solid #8a9a9266;border-radius:8px;
         padding:6px 12px;font-size:13px;font-weight:600;cursor:pointer}}
  button:hover{{border-color:#3ccf9f}}
  {dark_media}
</style>
<button id="b">📋 Copy answer</button>
<script>
  const text = {payload};
  const btn = document.getElementById('b');
  function fallback(){{
    const t = document.createElement('textarea'); t.value = text;
    document.body.appendChild(t); t.select();
    try {{ document.execCommand('copy'); }} catch (e) {{}}
    document.body.removeChild(t);
  }}
  btn.onclick = async () => {{
    try {{ await navigator.clipboard.writeText(text); }} catch (e) {{ fallback(); }}
    const old = btn.textContent; btn.textContent = '✓ Copied';
    setTimeout(() => btn.textContent = old, 1500);
  }};
</script>""",
        height=height,
    )


def doc_to_source(doc) -> dict:
    md = doc.metadata
    return {
        "file_path": md.get("file_path", md.get("file_name", "unknown")),
        "start_line": md.get("start_line"),
        "end_line": md.get("end_line"),
        "language": md.get("language", "text"),
        "score": md.get("rerank_score"),
        "content": doc.page_content,
    }


def render_sources(sources, grounded=True, expand_first=False):
    if not sources:
        return
    label = "📚 Sources" if grounded else "📚 Closest matches (below relevance threshold)"
    st.markdown(
        f'<div class="section-head" style="margin-top:14px;">{label} '
        f'<span class="chip">{len(sources)} snippets</span></div>',
        unsafe_allow_html=True,
    )
    for i, src in enumerate(sources, start=1):
        loc = f"{src['file_path']}:{src['start_line']}-{src['end_line']}"
        score = f"  ·  relevance {src['score']:.1f}" if src.get("score") is not None else ""
        with st.expander(f"[{i}] {loc}{score}", expanded=(expand_first and i == 1)):
            st.markdown(
                f'<span class="loc">{loc}</span> <span class="chip">{src["language"]}</span>',
                unsafe_allow_html=True,
            )
            st.code(src["content"], language=PRISM_LANG.get(src["language"], src["language"]))


def render_message(msg: dict, expand_sources: bool = False):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            copy_button(msg["content"])
            render_sources(msg.get("sources"), msg.get("grounded", True), expand_first=expand_sources)


STEPS = ["Add code", "Index", "Ask"]


def render_stepper(current: int) -> str:
    parts = []
    for i, label in enumerate(STEPS, start=1):
        state = "done" if i < current else ("active" if i == current else "")
        dot = "✓" if i < current else str(i)
        parts.append(
            f'<div class="step {state}"><div class="dot">{dot}</div><div class="step-label">{label}</div></div>'
        )
        if i < len(STEPS):
            parts.append(f'<div class="bar {"done" if i < current else ""}"></div>')
    return f'<div class="stepper">{"".join(parts)}</div>'


# ──────────────────────────────────────────────────────────────
# Sidebar: Ollama status, model picker, retrieval settings
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    status = cached_ollama_status()
    if status["ok"]:
        st.success("Ollama is running", icon="✅")
    else:
        st.error("Ollama is offline", icon="⚠️")
        st.caption("Start it with `ollama serve`, then refresh.")
    st.button("Refresh status", on_click=cached_ollama_status.clear, **STRETCH)

    models = status["models"]
    if models:
        idx = models.index(DEFAULT_MODEL) if DEFAULT_MODEL in models else 0
        st.selectbox("Model", models, index=idx, key="model")
    else:
        st.text_input("Model", value=DEFAULT_MODEL, key="model", help="No installed models detected.")

    with st.expander("Retrieval", expanded=False):
        st.selectbox("Mode", list(RETRIEVAL_MODES), format_func=RETRIEVAL_MODES.get, key="mode")
        st.slider(
            "Snippets to retrieve (top-k)",
            3,
            10,
            5,
            key="top_k",
            help="How many code chunks are sent to the LLM and shown as sources. "
                 "More = better recall but more noise and slower answers.",
        )
        st.slider(
            "Relevance cut-off (reranker score)",
            -12.0,
            4.0,
            float(min(max(MIN_RERANK_SCORE, -12.0), 4.0)),
            0.5,
            key="min_score",
            help="Chunks scoring below this are treated as irrelevant and the LLM is not called. "
                 "If you keep getting 'Information not found', lower it; "
                 "if answers look off-topic, raise it. Works in all retrieval modes.",
        )

    if st.session_state.step == 3:
        st.button(
            "Clear conversation",
            on_click=lambda: st.session_state.update(messages=[]),
            **STRETCH,
        )

# ──────────────────────────────────────────────────────────────
# Header + hero
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="nav">
    <div class="brand"><div class="brand-mark">🧠</div>RepoMind</div>
    <div class="nav-badge">🔒 Runs locally · Models and code processing stay on your machine</div>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown(render_stepper(st.session_state.step), unsafe_allow_html=True)

HERO = {
    1: ("Step 1 of 3", "Add your codebase", "Use a local folder, a single file, a .zip, or a public GitHub repo: Python, C#, JS, TS, Java, Go, Rust, C/C++, HTML/CSS, JSON/YAML/SQL and Markdown docs."),
    2: ("Step 2 of 3", "Indexing your code", "Parsing the AST and building a hybrid BM25 + vector index. Unchanged repos load from cache."),
    3: ("Step 3 of 3", "Ask anything about your code", "Grounded answers from local Ollama, with exact files and line ranges cited."),
}
eyebrow, title, subtitle = HERO[st.session_state.step]
st.markdown(
    f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# Main card
# ──────────────────────────────────────────────────────────────
with st.container(border=True):

    # STEP 1 — Repository selection
    if st.session_state.step == 1:
        st.markdown('<div class="card-title">Add your code</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-sub">Enter a local folder, a single file, a .zip, or a public GitHub URL.</div>',
            unsafe_allow_html=True,
        )

        source_input = st.text_input(
            "Folder, file, .zip path or GitHub URL",
            value=st.session_state.source_input,
            placeholder="e.g. ./my-project  ·  C:/Users/you/app.zip  ·  https://github.com/owner/repo",
        )
        uploaded = st.file_uploader(
            "…or upload a .zip or a single source file (takes priority over the box above)",
            type=["zip"] + sorted(ext.lstrip(".") for ext in LANGUAGE_CONFIG),
        )
        st.markdown(
            '<div class="chips">'
            + "".join(
                f'<span class="chip">{n}</span>'
                for n in ["Python", "C#", "JavaScript", "TypeScript", "Java", "Go", "Rust", "C++", "HTML", "CSS", "JSON", "YAML", "SQL", "Markdown"]
            )
            + "</div>",
            unsafe_allow_html=True,
        )

        if st.button("Continue to indexing →", type="primary"):
            resolved = None
            try:
                with st.spinner("Preparing your source (cloning / extracting if needed)…"):
                    resolved = resolve_source(source_input, uploaded)
            except SourceError as exc:
                st.error(str(exc))
            except Exception as exc:  # never let an unexpected loader error crash the page
                st.error(f"Could not load the source: {exc}")

            if resolved is not None:
                st.session_state.source_input = source_input
                st.session_state.repo_path = resolved.path
                st.session_state.source_label = resolved.label
                st.session_state.source_notes = resolved.notes
                st.session_state.step = 2
                st.rerun()

    # STEP 2 — Indexing (cached)
    elif st.session_state.step == 2:
        st.markdown('<div class="card-title">Reading &amp; vectorizing code…</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">Please keep this tab open while we build the index.</div>', unsafe_allow_html=True)

        error = None
        with st.status("Indexing…", expanded=True) as status_box:
            try:
                st.write("Scanning files…")
                fingerprint = compute_repo_fingerprint(st.session_state.repo_path, EXCLUDE_PATTERNS)
                st.write("Chunking by AST and building hybrid index (instant if unchanged)…")
                t0 = time.time()
                retriever, stats = build_index(st.session_state.repo_path, fingerprint)
                st.session_state.retriever, st.session_state.index_stats = retriever, stats
                status_box.update(
                    label=f"Indexed {stats['chunks']} chunks from {stats['files']} files in {time.time() - t0:.1f}s",
                    state="complete",
                    expanded=False,
                )
            except Exception as exc:
                error = exc
                status_box.update(label="Indexing failed", state="error")

        if error is None:
            st.session_state.step = 3
            time.sleep(0.4)
            st.rerun()
        else:
            st.error(f"Error during indexing: {error}")
            if st.button("← Try again"):
                st.session_state.step = 1
                st.rerun()

    # STEP 3 — Multi-turn Q&A
    elif st.session_state.step == 3:
        stats = st.session_state.index_stats or {}
        st.markdown('<div class="card-title">Ask your codebase</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-sub">Follow-up questions work — the conversation is remembered.</div>',
            unsafe_allow_html=True,
        )
        if stats:
            langs = "".join(f'<span class="chip">{l} · {n}</span>' for l, n in sorted(stats["languages"].items()))
            st.markdown(
                f'<div class="chips"><span class="chip brand">{html.escape(st.session_state.source_label or "source")}</span>'
                f'<span class="chip brand">{stats["files"]} files</span>'
                f'<span class="chip brand">{stats["chunks"]} chunks</span>{langs}</div>',
                unsafe_allow_html=True,
            )

        for note in st.session_state.get("source_notes", []):
            st.caption(note)

        ollama_ok = cached_ollama_status()["ok"]
        if not ollama_ok:
            st.warning(
                "Ollama isn't reachable, so questions are paused. Run `ollama serve` in a terminal, "
                "then press **Refresh status** in the sidebar.",
                icon="⚠️",
            )

        # Chat history renders into this container (above the form) — also while a new answer streams.
        history_box = st.container()

        cols = st.columns(len(SAMPLE_QUESTIONS))
        for col, sq in zip(cols, SAMPLE_QUESTIONS):
            col.button(
                sq,
                on_click=ask_sample,
                args=(sq,),
                key=f"sample-{sq}",
                disabled=not ollama_ok,
                **STRETCH,
            )

        # The form means typing / widget changes never trigger an LLM call; only "Ask" does.
        with st.form("ask_form", clear_on_submit=True):
            question = st.text_input(
                "Your question",
                key="question",
                placeholder="e.g. How does authentication work?",
            )
            submitted = st.form_submit_button("Ask", type="primary", disabled=not ollama_ok)

        with history_box:
            msgs = st.session_state.messages
            for i, m in enumerate(msgs):
                render_message(m, expand_sources=(i == len(msgs) - 1))

            pending = st.session_state.pop("pending_q", None)
            q = pending or (question.strip() if submitted and question else "")

            if q:
                live = ollama_health()  # fresh pre-flight check (not the 10s cache)
                model = st.session_state.get("model", DEFAULT_MODEL)
                installed = live["models"]

                if not live["ok"]:
                    st.warning("Ollama went offline. Start it with `ollama serve` and ask again.", icon="⚠️")
                elif installed and model not in installed and f"{model}:latest" not in installed:
                    st.warning(f"Model `{model}` isn't installed. Run `ollama pull {model}`.", icon="⚠️")
                else:
                    with st.chat_message("user"):
                        st.markdown(q)
                    with st.chat_message("assistant"):
                        with st.spinner("Searching your codebase…"):
                            result = stream_code_explanation(
                                q,
                                st.session_state.retriever,
                                model=model,
                                history=st.session_state.messages,  # cleaned inside the engine
                                top_k=st.session_state.get("top_k", 5),
                                mode=st.session_state.get("mode", "hybrid_rerank"),
                                min_score=st.session_state.get("min_score", MIN_RERANK_SCORE),
                            )
                        answer = st.write_stream(result.stream)
                        copy_button(answer)
                        sources = [doc_to_source(d) for d in result.docs]
                        render_sources(sources, result.grounded, expand_first=True)
                        if result.standalone_query != q:
                            st.caption(f"Searched for: _{result.standalone_query}_")

                    st.session_state.messages += [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": answer, "sources": sources, "grounded": result.grounded},
                    ]

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.button("← Index another codebase", on_click=reset_to_start)

st.markdown(
    '<div class="hint">RepoMind · Hybrid BM25 + vector search · Cross-encoder reranking · Local Ollama</div>',
    unsafe_allow_html=True,
)