import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import faiss
import json

from arxiv_explorer.process import process

MONGODB_URL = os.getenv("MONGODB_URL")


def test_e2e_process():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension()))
    mongo_client = MongoClient(MONGODB_URL)
    mongo_collection = mongo_client["arxivdb"]["arxivcol"]
    dataset_path = "./tests/e2e/data/arxiv-metadata-oai-sample.json"

    _ = process(
        dataset_path=dataset_path,
        embedding_model=embedding_model,
        faiss_index=faiss_index,
        mongo_collection=mongo_collection,
        num_to_proces=50,
        attribute_to_encode="abstract",
    )
    with open(dataset_path, "r") as f:
        file_entries = [json.loads(line) for line in f]

    for i, (document, file_entry) in enumerate(zip(mongo_collection.find(), file_entries)):
        file_entry["faiss_id"] = i
        assert file_entry == {k:v for k, v in document.items() if k != "_id"}

    # test all ids are in saved faiss index
    assert len(file_entries) == faiss_index.ntotal
    faiss_ids = faiss.vector_to_array(faiss_index.id_map).tolist()
    for i in range(len(file_entries)):
        assert faiss_ids[i] == i
