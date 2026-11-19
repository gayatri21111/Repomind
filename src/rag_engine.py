"""
Generation: grounded, multi-turn code Q&A on top of a local Ollama model.

Anti-hallucination layers
-------------------------
1. Retrieval gate  - if the best cross-encoder score is below MIN_RERANK_SCORE the
                     LLM is never called; the canned NOT_FOUND_MESSAGE is returned.
                     The gate now works in EVERY retrieval mode: modes that do not
                     rerank get their chunks scored (order untouched) before the gate.
2. Strict system prompt - answer only from numbered snippets, cite [n] + file:lines,
                     otherwise return the NOT_FOUND_MESSAGE.
3. Follow-ups are first rewritten into standalone queries so retrieval sees
   "Where is `build_index` called?" rather than "Where is that called?".
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence

import requests
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

try:
    from langchain_ollama import ChatOllama
except ImportError:  # pragma: no cover
    from langchain_community.chat_models import ChatOllama

DEFAULT_MODEL = "llama3.2:3b"
NOT_FOUND_MESSAGE = "Information not found in codebase."
# ms-marco cross-encoder logits: clearly irrelevant pairs sit around -10; tune with evaluate.py
# or with the sidebar slider in the app.
MIN_RERANK_SCORE = float(os.environ.get("REPOMIND_MIN_RERANK", "-6.0"))
MAX_HISTORY_TURNS = 6
MAX_HISTORY_CHARS = 2000  # per message, keeps history from eating the context window
NUM_CTX = 8192  # Ollama defaults to 2048, which silently truncates retrieved context
ERROR_MARK = "⚠️"


def _host() -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").strip().rstrip("/")
    return host if host.startswith("http") else f"http://{host}"


# ──────────────────────────────────────────────────────────────
# Ollama pre-flight
# ──────────────────────────────────────────────────────────────
def ollama_health(timeout: float = 2.0) -> Dict:
    """Return {'ok': bool, 'models': [names], 'error': str|None}."""
    try:
        resp = requests.get(f"{_host()}/api/tags", timeout=timeout)
        resp.raise_for_status()
        models = sorted(m.get("name") or m.get("model", "") for m in resp.json().get("models", []))
        return {"ok": True, "models": [m for m in models if m], "error": None}
    except Exception as exc:
        return {"ok": False, "models": [], "error": str(exc)}


def list_ollama_models() -> List[str]:
    return ollama_health()["models"]


# ──────────────────────────────────────────────────────────────
# Prompting
# ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are RepoMind, a senior software engineer answering questions about a codebase.

Rules:
1. Use ONLY the numbered code snippets inside <context>. Never use outside knowledge about this project, and never invent files, functions, classes, parameters or behaviour.
2. If the snippets do not contain enough information to answer, reply with exactly this sentence and nothing else: {NOT_FOUND_MESSAGE}
3. Cite the evidence for every claim as [n] followed by file path and line range, for example [2] `src/auth.py:10-42`.
4. Only put code in fenced blocks if it appears verbatim in the snippets.
5. Earlier conversation turns may be used to understand references such as "that function", but facts must still come from <context>.
6. Be concise and precise."""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT + "\n\n<context>\n{context}\n</context>"),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

CONDENSE_PROMPT = ChatPromptTemplate.from_template(
    """Rewrite the follow-up question as one standalone question for searching a codebase.
Resolve pronouns and references using the conversation. Keep identifiers, file names and
function names exactly as written. Output ONLY the rewritten question.

Conversation:
{history}

Follow-up question: {question}

Standalone question:"""
)


def _llm(model: str, temperature: float = 0.1) -> ChatOllama:
    return ChatOllama(model=model, temperature=temperature, base_url=_host(), num_ctx=NUM_CTX)


def usable_history(messages: Optional[Sequence[Dict]]) -> List[Dict]:
    """
    Keep only clean (user, assistant) pairs for conversation memory.
    Canned "not found" replies and "⚠️ Ollama request failed" errors are dropped so
    they never pollute follow-up questions or the query rewriter.
    """
    clean: List[Dict] = []
    msgs = list(messages or [])
    for i in range(0, len(msgs) - 1, 2):
        user, assistant = msgs[i], msgs[i + 1]
        if user.get("role") != "user" or assistant.get("role") != "assistant":
            continue
        if not assistant.get("grounded", True):
            continue
        content = assistant.get("content", "")
        if ERROR_MARK in content or content.strip() == NOT_FOUND_MESSAGE:
            continue
        clean += [
            {"role": "user", "content": user.get("content", "")},
            {"role": "assistant", "content": content},
        ]
    return clean


def history_to_messages(history: Optional[Sequence[Dict]], max_turns: int = MAX_HISTORY_TURNS) -> List[BaseMessage]:
    """Convert [{'role','content'}] session history into LangChain messages (last N turns)."""
    messages: List[BaseMessage] = []
    for item in list(history or [])[-2 * max_turns:]:
        cls = HumanMessage if item.get("role") == "user" else AIMessage
        messages.append(cls(content=item.get("content", "")[:MAX_HISTORY_CHARS]))
    return messages


_PREFIX = re.compile(r"^\s*(standalone question\s*:|rewritten question\s*:|question\s*:)\s*", re.I)


def _clean_rewrite(text: str) -> str:
    """Small local models like to add labels, quotes or extra lines. Keep just the question."""
    text = text.strip().splitlines()[0] if text.strip() else ""
    text = _PREFIX.sub("", text).strip().strip('"').strip("'").strip()
    return text


def condense_question(question: str, history: Optional[Sequence[Dict]], model: str) -> str:
    """Turn a follow-up into a standalone retrieval query. Falls back to the raw question."""
    if not history:
        return question
    transcript = "\n".join(
        f"{m.get('role', 'user').capitalize()}: {m.get('content', '')[:500]}" for m in list(history)[-4:]
    )
    try:
        out = _clean_rewrite(
            (CONDENSE_PROMPT | _llm(model, 0.0) | StrOutputParser()).invoke(
                {"history": transcript, "question": question}
            )
        )
        return out if 0 < len(out) <= 500 else question
    except Exception:
        return question


def format_docs(docs: Sequence[Document]) -> str:
    """Number snippets and attach file:line provenance so the LLM can cite them."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        md = doc.metadata
        path = md.get("file_path") or md.get("file_name", "unknown")
        loc = f"{path}:{md.get('start_line', '?')}-{md.get('end_line', '?')}"
        lang = md.get("language", "")
        blocks.append(f"[{i}] {loc}\n```{lang}\n{doc.page_content}\n```")
    return "\n\n".join(blocks)


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────
@dataclass
class RagResult:
    stream: Iterator[str]
    docs: List[Document]
    standalone_query: str
    grounded: bool  # False -> retrieval gate tripped (or retrieval failed), LLM was not called


def _ensure_scores(retriever, query: str, docs: List[Document]) -> List[Document]:
    """
    Only the reranker mode produces a relevance score. For the other modes, score the
    already-retrieved chunks (order is preserved) so the cut-off slider has an effect.
    """
    if not docs or any("rerank_score" in d.metadata for d in docs):
        return docs
    scorer = getattr(retriever, "score_docs", None)
    if scorer is None:
        return docs
    try:
        return scorer(query, docs)
    except Exception:
        return docs  # scoring is best-effort; fall back to ungated behaviour


def _is_relevant(docs: Sequence[Document], min_score: float) -> bool:
    if not docs:
        return False
    scores = [d.metadata["rerank_score"] for d in docs if "rerank_score" in d.metadata]
    return True if not scores else max(scores) >= min_score


def _safe_stream(chain, inputs: Dict) -> Iterator[str]:
    try:
        yield from chain.stream(inputs)
    except Exception as exc:
        yield f"\n\n{ERROR_MARK} Ollama request failed: {exc}"


def stream_code_explanation(
    query: str,
    retriever,
    model: str = DEFAULT_MODEL,
    history: Optional[Sequence[Dict]] = None,
    top_k: int = 5,
    mode: str = "hybrid_rerank",
    min_score: float = MIN_RERANK_SCORE,
) -> RagResult:
    """Retrieve (eagerly) and return a lazy token stream plus the evidence used."""
    history = usable_history(history)
    standalone = condense_question(query, history, model)

    try:
        docs = retriever.retrieve(standalone, mode=mode, top_k=top_k)
        docs = _ensure_scores(retriever, standalone, docs)
    except Exception as exc:  # a retrieval error must not crash the whole Streamlit run
        return RagResult(iter([f"{ERROR_MARK} Retrieval failed: {exc}"]), [], standalone, grounded=False)

    if not _is_relevant(docs, min_score):
        return RagResult(iter([NOT_FOUND_MESSAGE]), docs, standalone, grounded=False)

    # The gate above only checks the BEST chunk. Also drop chunks that individually fall below
    # the cut-off, so small repos don't send (and show) the same unrelated snippets every time.
    if any("rerank_score" in d.metadata for d in docs):
        docs = [d for d in docs if d.metadata.get("rerank_score", min_score) >= min_score]

    chain = QA_PROMPT | _llm(model) | StrOutputParser()
    inputs = {
        "context": format_docs(docs),
        "history": history_to_messages(history),
        "question": query,
    }
    return RagResult(_safe_stream(chain, inputs), docs, standalone, grounded=True)