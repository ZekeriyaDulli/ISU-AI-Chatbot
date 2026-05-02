"""
Unit Tests — Vector DB & RAG Pipeline
Evaluation & Testing Lead: Leen Safi (STU ID: 2309116117)
"""

import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from rag.vector_db import VectorDBClient, DocumentIngestionPipeline


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma_test"))
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "test_collection")
    return VectorDBClient()


@pytest.fixture
def pipeline(db):
    return DocumentIngestionPipeline(db, chunk_size=50, overlap=10)


def test_empty_db_count(db):
    assert db.count == 0


def test_ingest_and_count(db):
    added = db.ingest(["Hello world", "Climate change is real."])
    assert added == 2
    assert db.count == 2


def test_query_returns_results(db):
    db.ingest(["The sky is blue because of Rayleigh scattering."])
    results = db.query("Why is the sky blue?")
    assert len(results) > 0
    assert isinstance(results[0], str)


def test_empty_query_returns_empty(db):
    results = db.query("anything")
    assert results == []


def test_pipeline_chunking(pipeline, db):
    long_text = " ".join(["word"] * 200)
    added = pipeline.ingest_text(long_text, source="test")
    assert added > 1
    assert db.count == added
