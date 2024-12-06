import json
import numpy as np
import faiss
from pymongo import MongoClient
from pymongo.collection import Collection
import logging
from tqdm import tqdm
import argparse
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

dataset_path = "/Users/tomcarter/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/207/arxiv-metadata-oai-snapshot.json"


class DataHandler:

    def __init__(
        self,
        dataset: str,
        embedding_model: SentenceTransformer,
        faiss_index: faiss.Index,
        mongo_collection: Collection,
        num_to_proces: int = -1,
    ):
        self.dataset_path = dataset
        self.embedding_model = embedding_model
        self.faiss_index = faiss_index
        self.num_to_proces = num_to_proces
        self.mongo_collection = mongo_collection

    @staticmethod
    def load_data(dataset: str, n: int = -1):

        if n == 0:
            raise ValueError("n must be greater than 0, or -1 to process all documents")

        with open(dataset) as f:
            for i, line in enumerate(f):
                if i >= n > 0:
                    break
                yield json.loads(line)

    def get_embedding_vector(self, data: str) -> np.ndarray:
        embedding = self.embedding_model.encode(data)
        return np.array([embedding])

    def process_one(self, input_data: dict, i: int, attribute_to_encode: str = "abstract") -> None:
        embedding = self.get_embedding_vector(input_data[attribute_to_encode])
        self.faiss_index.add_with_ids(embedding, np.array([i]))
        input_data["faiss_id"] = i
        print("inserting mongo")
        self.mongo_collection.insert_one(input_data)

    def process_and_save(self, index_path: str, attribute_to_encode: str = "abstract") -> None:
        logger.info("Beginning pipeline")

        for i, input_data in enumerate(tqdm(self.load_data(self.dataset_path, n=self.num_to_proces))):
            self.process_one(input_data, i, attribute_to_encode)

        faiss.write_index(self.faiss_index, index_path)
        logger.info(f"Done")


def setup_logging(level: str):
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s\t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)


def main() -> None:
    """Runs the script."""
    parser = argparse.ArgumentParser(
        description="Run processing pipeline to save embeddings and mongodb."
    )
    parser.add_argument(
        "--log",
        default="WARNING",
        help=f"Set the logging level. Available options: {list(logging._nameToLevel.keys())}",
    )
    parser.add_argument(
        "-n",
        default="-1",
        help=f"Number of documents to process. Default is -1, which processes all documents.",
    )

    args = parser.parse_args()

    setup_logging(level=args.log)

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension()))
    mongo_client = MongoClient("mongodb://localhost:27017/")
    mongo_collection = mongo_client["arxivdb"]["arxivcol"]

    dh = DataHandler(
        dataset=dataset_path,
        embedding_model=embedding_model,
        faiss_index=faiss_index,
        mongo_collection=mongo_collection,
        num_to_proces=int(args.n),
    )
    dh.process_and_save(index_path="./faiss_index.faiss")


if __name__ == "__main__":
    main()
