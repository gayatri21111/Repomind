"""
Retrieval: ChromaDB (semantic) + BM25 (lexical) -> Reciprocal Rank Fusion
-> CrossEncoder re-ranking.

Modes (see RETRIEVAL_MODES) let the same index be queried four ways, which is
what evaluate.py uses to benchmark each stage.

Changes vs. the previous version
--------------------------------
* `score_docs()` scores any list of chunks with the cross-encoder WITHOUT
  re-ordering them. rag_engine uses it so the relevance cut-off also works in
  the "vector", "bm25" and "hybrid" modes (before, it only worked with the
  reranker mode because only that mode produced a score).
* The BM25 tokenizer drops English stop-words ("the", "is", "where" ...). They
  carried no signal and BM25 returns the top-k chunks even when nothing matches.
* `top_k` is validated.
"""
from __future__ import annotations

import re
import threading
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

try:
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover
    from langchain_community.vectorstores import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover
    from langchain_community.embeddings import HuggingFaceEmbeddings

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VECTOR_DB_PATH = "./chroma_db"
RRF_K = 60
FETCH_K = 20       # candidates pulled from each retriever before fusion
RERANK_POOL = 30   # max fused candidates sent to the cross-encoder

RETRIEVAL_MODES: Dict[str, str] = {
    "hybrid_rerank": "Hybrid (RRF) + reranker",
    "hybrid": "Hybrid (RRF)",
    "vector": "Vector only",
    "bm25": "BM25 only",
}


# ──────────────────────────────────────────────────────────────
# Models (lazy singletons)
# ──────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL, max_length=512)


# ──────────────────────────────────────────────────────────────
# Keyword search helpers
# ──────────────────────────────────────────────────────────────
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Natural-language filler only. Deliberately small so real identifiers survive.
_STOPWORDS = frozenset(
    "a an the is are was were be been being of in on at to and or but how what where which who whom "
    "when why does do did this that these those it its with for from by as can could should would "
    "me my i you your we our please tell explain show give about there here".split()
)


def code_tokenizer(text: str) -> List[str]:
    """
    Identifier-aware tokenizer for BM25.
    `loadUserProfile` -> [loaduserprofile, load, user, profile]
    `max_chunk_size`  -> [max_chunk_size, max, chunk, size]
    `__init__`        -> [__init__, init]
    so natural-language queries can hit identifiers and vice versa.
    Stop-words are removed; if that would leave nothing, the unfiltered tokens are used.
    """
    tokens: List[str] = []
    for word in _WORD.findall(text):
        lowered = word.lower()
        tokens.append(lowered)
        parts = [p.lower() for p in re.split(r"_+", _CAMEL_BOUNDARY.sub("_", word)) if p]
        if parts and parts != [lowered]:
            tokens.extend(parts)
    filtered = [t for t in tokens if t not in _STOPWORDS]
    return filtered or tokens


def _doc_key(doc: Document) -> str:
    return doc.metadata.get("chunk_id") or f"{doc.metadata.get('source', '')}_{hash(doc.page_content)}"


def rrf_fuse(rankings: Sequence[Sequence[Document]], weights: Sequence[float], k: int = RRF_K) -> List[Document]:
    """Weighted Reciprocal Rank Fusion: score = sum_i w_i / (k + rank_i)."""
    scores: Dict[str, float] = {}
    docs: Dict[str, Document] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, doc in enumerate(ranking, start=1):
            key = _doc_key(doc)
            docs[key] = doc
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)]


# ──────────────────────────────────────────────────────────────
# Retriever
# ──────────────────────────────────────────────────────────────
class HybridRetriever:
    """Vector + BM25 -> RRF -> optional cross-encoder reranking."""

    def __init__(
        self,
        vectorstore,
        bm25: BM25Retriever,
        weights: Sequence[float] = (0.5, 0.5),   # (bm25, vector)
        fetch_k: int = FETCH_K,
        default_top_k: int = 5,
        default_mode: str = "hybrid_rerank",
    ):
        self.vectorstore = vectorstore
        self.bm25 = bm25
        self.weights = tuple(weights)
        self.fetch_k = fetch_k
        self.default_top_k = default_top_k
        self.default_mode = default_mode
        self._bm25_lock = threading.Lock()  # retriever is shared across Streamlit sessions

    # -- individual stages ---------------------------------------------------
    def _vector_search(self, query: str, n: int) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=n)

    def _bm25_search(self, query: str, n: int) -> List[Document]:
        with self._bm25_lock:
            self.bm25.k = n
            return self.bm25.invoke(query)

    @staticmethod
    def _cross_scores(query: str, docs: Sequence[Document]) -> List[float]:
        pairs = [(query, f"{d.metadata.get('file_path', '')}\n{d.page_content}") for d in docs]
        return [float(s) for s in get_reranker().predict(pairs)]

    def rerank(self, query: str, docs: Sequence[Document], top_k: int) -> List[Document]:
        """Score (query, chunk) pairs with the cross-encoder; keep the best `top_k`."""
        if not docs:
            return []
        scores = self._cross_scores(query, docs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            Document(page_content=d.page_content, metadata={**d.metadata, "rerank_score": s})
            for d, s in ranked
        ]

    def score_docs(self, query: str, docs: Sequence[Document]) -> List[Document]:
        """
        Attach a cross-encoder `rerank_score` to each chunk but KEEP the existing order.
        Lets the relevance cut-off work for vector / bm25 / hybrid modes too.
        """
        if not docs:
            return []
        scores = self._cross_scores(query, docs)
        return [
            Document(page_content=d.page_content, metadata={**d.metadata, "rerank_score": s})
            for d, s in zip(docs, scores)
        ]

    # -- public API ----------------------------------------------------------
    def retrieve(self, query: str, mode: Optional[str] = None, top_k: Optional[int] = None) -> List[Document]:
        mode = mode or self.default_mode
        top_k = max(1, int(top_k or self.default_top_k))
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {list(RETRIEVAL_MODES)}")

        if mode == "vector":
            return self._vector_search(query, top_k)
        if mode == "bm25":
            return self._bm25_search(query, top_k)

        n = max(self.fetch_k, top_k)
        fused = rrf_fuse([self._bm25_search(query, n), self._vector_search(query, n)], self.weights)
        if mode == "hybrid":
            return fused[:top_k]
        return self.rerank(query, fused[:RERANK_POOL], top_k)

    def invoke(self, query: str) -> List[Document]:
        return self.retrieve(query)

    get_relevant_documents = invoke  # backwards compatibility

    def warmup(self) -> None:
        """Load the embedding model and cross-encoder so the first question is fast."""
        self._vector_search("warmup", 1)
        get_reranker()


def build_hybrid_retriever(
    chunks: List[Document],
    k: int = 5,
    fingerprint: Optional[str] = None,
    persist_directory: str = VECTOR_DB_PATH,
) -> HybridRetriever:
    """
    Index chunks into Chroma (persistent, one collection per repo fingerprint)
    and BM25 (in-memory), and return a HybridRetriever.
    A matching persisted collection is reused instead of re-embedding.

    Note: every fingerprint gets its own collection, so old collections stay in
    ./chroma_db after files change. Delete the folder occasionally to reclaim space.
    """
    if not chunks:
        raise ValueError("Cannot index empty document list.")

    collection = f"repomind_{fingerprint[:24]}" if fingerprint else "repomind_default"

    def _open():
        return Chroma(collection_name=collection, embedding_function=get_embeddings(), persist_directory=persist_directory)

    vectorstore = _open()
    existing = vectorstore._collection.count()  # public count API is not exposed by the wrapper
    if fingerprint is None or existing != len(chunks):
        if existing:
            vectorstore.delete_collection()
            vectorstore = _open()
        ids = [c.metadata["chunk_id"] for c in chunks]
        for i in range(0, len(chunks), 256):
            vectorstore.add_documents(chunks[i:i + 256], ids=ids[i:i + 256])

    bm25 = BM25Retriever.from_documents(chunks, preprocess_func=code_tokenizer)
    bm25.k = k
    return HybridRetriever(vectorstore, bm25, default_top_k=k)