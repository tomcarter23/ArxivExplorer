from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from arxiv_explorer.retrieve import Retriever


class MockSentenceTransformer:
    encode = MagicMock(return_value=[1, 2, 3])


class MockFaissIndex:
    add_with_ids = MagicMock()
    write_index = MagicMock()
    search = MagicMock(return_value=(None, np.array([[1, 2, 3]])))


class MockMongoCollection:
    insert_one = MagicMock()
    find = MagicMock(return_value=[{
        "faiss_id": 1,
        "title": "The Art of Computer Programming",
        "abstract": "This is a book about the C programming language",
    }])


@pytest.fixture
@patch("pymongo.collection.Collection", return_value=MockMongoCollection())
@patch("faiss.Index", return_value=MockFaissIndex())
@patch("sentence_transformers.SentenceTransformer", return_value=MockSentenceTransformer())
def mock_retriever(mock_sentence_transformer, mock_faiss_index, mock_mongo_collection):
    retriever = Retriever(
        embedding_model=mock_sentence_transformer(),
        faiss_index=mock_faiss_index(),
        mongo_collection=mock_mongo_collection(),
    )
    return retriever


def test_retriever_fetch_document_by_id(mock_retriever):
    doc = mock_retriever.fetch_document_by_id(1)
    assert doc["title"] == "The Art of Computer Programming"
    assert doc["abstract"] == "This is a book about the C programming language"


@patch("arxiv_explorer.retrieve.Retriever.fetch_document_by_id", return_value={
    "title": "The Art of Computer Programming",
    "abstract": "This is a book about the C programming language",
    })
def test_retriever_retrieve(mock_fetch_document_by_id, mock_retriever):
    docs = [d for d in mock_retriever.retrieve(prompt="The Art of Computer Programming", k=3)]
    assert len(docs) == 3
    assert docs[0]["title"] == "The Art of Computer Programming"
    assert docs[0]["abstract"] == "This is a book about the C programming language"
