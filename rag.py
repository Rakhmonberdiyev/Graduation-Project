import io
import os
import uuid

import pypdf
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

load_dotenv()

COLLECTION    = "tg_rag_documents"
EMBED_MODEL   = "text-embedding-3-small"
VECTOR_SIZE   = 1536
CHUNK_WORDS   = 400
OVERLAP_WORDS = 60
RAG_THRESHOLD = 0.10

_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_qdrant = QdrantClient(host="localhost", port=int(os.getenv("QDRANT_PORT", "6333")))


def _ensure_collection():
    names = [c.name for c in _qdrant.get_collections().collections]
    if COLLECTION not in names:
        _qdrant.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def _chunk(text: str) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + CHUNK_WORDS]))
        i += CHUNK_WORDS - OVERLAP_WORDS
    return [c for c in chunks if c.strip()]


def _embed(texts: list[str]) -> list[list[float]]:
    resp = _openai.embeddings.create(model=EMBED_MODEL, input=texts)
    return [r.embedding for r in resp.data]


def _parse_pdf(data: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _parse_txt(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore")


# ── Public API ─────────────────────────────────────────────────────────────────

def store_document(file_bytes: bytes, mime_type: str, filename: str, user_id: str) -> dict:
    """Parse, chunk, embed and store a document. Returns stats dict."""
    _ensure_collection()

    if "pdf" in mime_type:
        text = _parse_pdf(file_bytes)
    else:
        text = _parse_txt(file_bytes)

    chunks = _chunk(text)
    if not chunks:
        return {"chunks": 0, "chars": 0, "filename": filename}

    embeddings = _embed(chunks)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={
                "user_id":     user_id,
                "filename":    filename,
                "chunk":       chunk,
                "chunk_index": i,
            },
        )
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    _qdrant.upsert(collection_name=COLLECTION, points=points)
    return {"chunks": len(chunks), "chars": len(text), "filename": filename}


def search_documents(
    query: str,
    user_id: str,
    filename: str | None = None,
    top_k: int = 5,
) -> tuple[list[dict], list[dict]]:
    """
    Returns (hits_above_threshold, all_raw_hits).
    If filename is given, search is restricted to that file only.
    """
    _ensure_collection()
    [q_emb] = _embed([query])

    must_conditions = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
    if filename:
        must_conditions.append(FieldCondition(key="filename", match=MatchValue(value=filename)))

    result = _qdrant.query_points(
        collection_name=COLLECTION,
        query=q_emb,
        query_filter=Filter(must=must_conditions),
        limit=top_k,
        with_payload=True,
    )
    all_hits = [
        {
            "chunk":    h.payload["chunk"],
            "filename": h.payload["filename"],
            "score":    round(h.score, 4),
        }
        for h in result.points
    ]
    hits = [h for h in all_hits if h["score"] >= RAG_THRESHOLD]
    return hits, all_hits


def has_documents(user_id: str) -> bool:
    _ensure_collection()
    points, _ = _qdrant.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        ),
        limit=1,
    )
    return len(points) > 0


def list_documents(user_id: str) -> list[str]:
    """Return unique filenames uploaded by this user."""
    _ensure_collection()
    all_points, _ = _qdrant.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        ),
        limit=1000,
        with_payload=True,
    )
    seen, names = set(), []
    for p in all_points:
        fn = p.payload.get("filename", "")
        if fn not in seen:
            seen.add(fn)
            names.append(fn)
    return names


def delete_documents(user_id: str):
    """Delete all document chunks for a user."""
    _ensure_collection()
    _qdrant.delete(
        collection_name=COLLECTION,
        points_selector=FilterSelector(filter=Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )),
    )
