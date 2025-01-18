import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from arxiv_explorer.process import DataHandler


class MockSentenceTransformer:
    encode = MagicMock(return_value=[1, 2, 3])


class MockFaissIndex:
    add_with_ids = MagicMock()
    write_index = MagicMock()


class MockMongoCollection:
    insert_one = MagicMock()


@pytest.fixture
def data_path():
    return "./tests/unit/data/paper_data.json"


@pytest.fixture(autouse=True)
def reset_mocks():
    MockSentenceTransformer.encode.reset_mock()
    MockFaissIndex.add_with_ids.reset_mock()
    MockMongoCollection.insert_one.reset_mock()
    MockFaissIndex.write_index.reset_mock()


@pytest.fixture
@patch("pymongo.collection.Collection", return_value=MockMongoCollection())
@patch("faiss.Index", return_value=MockFaissIndex())
@patch("sentence_transformers.SentenceTransformer", return_value=MockSentenceTransformer())
def mock_data_handler(mock_sentence_transformer, mock_faiss_index, mock_mongo_collection, data_path):
    dh = DataHandler(
        dataset=data_path,
        embedding_model=mock_sentence_transformer(),
        faiss_index=mock_faiss_index(),
        mongo_collection=mock_mongo_collection(),
    )
    return dh


def test_data_handler_load_data(mock_data_handler, data_path):
    data = [d for d in mock_data_handler.load_data(data_path)]
    assert len(data) == 2
    assert data[0]["title"] == "The Art of Computer Programming"
    assert data[1]["abstract"] == "This is a book about the C programming language"


def test_data_handler_load_data_n(mock_data_handler, data_path):
    data = [d for d in mock_data_handler.load_data(data_path, n=1)]
    assert len(data) == 1
    assert data[0]["title"] == "The Art of Computer Programming"


def test_data_handler_load_data_n0_raises(mock_data_handler, data_path):
    with pytest.raises(ValueError):
        _ = [d for d in mock_data_handler.load_data(data_path, n=0)]


def test_data_handler_embedding_vector(mock_data_handler):
    embedding = mock_data_handler.get_embedding_vector("This is a test")
    assert (embedding == np.array([[1, 2, 3]])).all()


def test_data_handler_process_one(mock_data_handler):
    mock_data_handler.process_one(
        input_data={"abstract": "This is a test"},
        i=0,
        attribute_to_encode="abstract"
    )
    add_with_ids_call_args = mock_data_handler.faiss_index.add_with_ids.call_args
    embedding, idx = add_with_ids_call_args[0][0], add_with_ids_call_args[0][1]
    assert (embedding == np.array([[1, 2, 3]])).all()
    assert idx == np.array([0])

    insert_one_call_args = mock_data_handler.mongo_collection.insert_one.call_args
    insert = insert_one_call_args[0][0]
    assert insert == {"abstract": "This is a test", "faiss_id": 0}


def test_data_handler_process_and_save(mock_data_handler):
    with patch("faiss.write_index") as mock_write_index:
        mock_data_handler.process(attribute_to_encode="abstract")
        assert mock_data_handler.mongo_collection.insert_one.call_count == 2
        assert mock_data_handler.faiss_index.add_with_ids.call_count == 2
