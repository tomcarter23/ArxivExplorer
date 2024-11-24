import json
import numpy as np
import faiss
from pymongo import MongoClient
import logging
from tqdm import tqdm
import argparse
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

mongo_client = MongoClient("mongodb://localhost:27017/")

dataset_path = "/Users/tomcarter/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/207/arxiv-metadata-oai-snapshot.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class DataHandler:

    def __init__(self, dataset_path: str, embedding_model: str):
        self.dataset_path = dataset_path
        self.embedding_model = SentenceTransformer(embedding_model)
        vector_dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(vector_dimension))

        self.mongo_db = mongo_client["arxivdb"]
        self.mongo_collection = self.mongo_db["arxivcol"]

    def get_input(self, dataset_path: str): 
        with open(dataset_path) as f:
            for i, line in enumerate(f):
                if i > 10000: 
                    break
                yield json.loads(line)

    def get_embedding_vector(self, data: str):
        embedding = self.embedding_model.encode(data)
        return np.array([embedding])

    def add_embedding_to_faiss(self, embedding: np.ndarray, id: int): 
        self.faiss_index.add_with_ids(embedding, np.array([id]))

    def process_and_save(self, index_path: str):
        logger.info("Beginning pipeline")
        for i, input_data in enumerate(tqdm(self.get_input(self.dataset_path))):
            embedding = self.get_embedding_vector(input_data["abstract"])
            self.add_embedding_to_faiss(embedding=embedding, id=i)
            input_data["faiss_id"] = i
            self.mongo_collection.insert_one(input_data)
        faiss.write_index(self.faiss_index, index_path)
        logger.info(f"Done")



def setup_logging(level: str):
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s\t%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)


def main() -> None:
    """Runs the script."""
    parser = argparse.ArgumentParser(
        description="Run processing pipeline to save embeddings and mongodb."
    )
    parser.add_argument(
        "--log", default="WARNING", help=f"set the logging level. Available options: {list(logging._nameToLevel.keys())}"
    )

    args = parser.parse_args()

    setup_logging(level=args.log)

    dh = DataHandler(dataset_path=dataset_path, embedding_model=EMBEDDING_MODEL)
    dh.process_and_save(index_path="./faiss_index.faiss")


if __name__ == "__main__":
    main()
