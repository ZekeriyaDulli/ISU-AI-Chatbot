"""
Vector Database Client — ChromaDB + Sentence-Transformer Embeddings
Data & RAG Pipeline Engineer: Fares STOUHI (STU ID: 2309115179)
"""

import os
from typing import Optional
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


class VectorDBClient:
    """
    Wraps ChromaDB for document ingestion and semantic retrieval.
    Uses SentenceTransformer embeddings for high-dimensional vector indexing.
    """

    def __init__(self):
        # Read at construction time so monkeypatch works in tests
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        collection_name = os.getenv("CHROMA_COLLECTION_NAME", "knowledge_base")

        self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.encode(texts, normalize_embeddings=True).tolist()

    def ingest(
        self,
        documents: list[str],
        ids: Optional[list[str]] = None,
        metadatas: Optional[list[dict]] = None,
    ) -> int:
        """Add documents to the collection. Returns number of docs added."""
        if not ids:
            existing = self._collection.count()
            ids = [f"doc_{existing + i}" for i in range(len(documents))]

        embeddings = self._embed(documents)
        # ChromaDB ≥1.5 rejects empty metadata dicts — use None when no metadata provided
        clean_meta = [m if m else None for m in (metadatas or [None] * len(documents))]
        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=clean_meta,
        )
        return len(documents)

    def query(self, query_text: str, n_results: int = 4) -> list[str]:
        """Retrieve top-n semantically similar documents for a query."""
        if self._collection.count() == 0:
            return []

        embedding = self._embed([query_text])
        results = self._collection.query(
            query_embeddings=embedding,
            n_results=min(n_results, self._collection.count()),
        )
        return results["documents"][0] if results["documents"] else []

    def delete_collection(self) -> None:
        self._client.delete_collection(self._collection.name)

    @property
    def count(self) -> int:
        return self._collection.count()


class DocumentIngestionPipeline:
    """
    Handles chunking and ingestion of raw text documents into the vector store.
    Chunking strategy: fixed-size with overlap for context preservation.
    """

    def __init__(self, db: VectorDBClient, chunk_size: int = 512, overlap: int = 64):
        self.db = db
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _chunk(self, text: str) -> list[str]:
        words = text.split()
        chunks, i = [], 0
        while i < len(words):
            chunk = " ".join(words[i : i + self.chunk_size])
            chunks.append(chunk)
            i += self.chunk_size - self.overlap
        return chunks

    def ingest_text(self, text: str, source: str = "unknown") -> int:
        chunks = self._chunk(text)
        metadatas = [{"source": source, "chunk_index": idx} for idx, _ in enumerate(chunks)]
        return self.db.ingest(chunks, metadatas=metadatas)

    def ingest_file(self, filepath: str) -> int:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        return self.ingest_text(text, source=filepath)


if __name__ == "__main__":
    db = VectorDBClient()
    pipeline = DocumentIngestionPipeline(db)

    sample = (
        "Climate change refers to long-term shifts in global temperatures and weather patterns. "
        "Human activities, primarily the burning of fossil fuels, have been the main driver "
        "of climate change since the 1800s. This releases greenhouse gases like CO2 and methane."
    )
    added = pipeline.ingest_text(sample, source="sample_doc")
    print(f"Ingested {added} chunk(s). Total in DB: {db.count}")

    results = db.query("What causes climate change?")
    for i, doc in enumerate(results):
        print(f"\n[Result {i+1}]: {doc[:200]}...")
