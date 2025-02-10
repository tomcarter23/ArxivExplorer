import faiss
import numpy as np
from pymongo import MongoClient
from pymongo.collection import Collection
import argparse
import logging
import typing as ty

from sentence_transformers import SentenceTransformer
from .logging_utils import setup_logging

logger = logging.getLogger(__name__)
setup_logging(level="INFO", logger=logger)


class Retriever:

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        faiss_index: faiss.Index,
        mongo_collection: Collection,
    ):
        self.embedding_model = embedding_model
        self.faiss_index = faiss_index
        self.mongo_collection = mongo_collection
        logger.info("Retriever initialized")

    def retrieve(self, prompt: str, k: int) -> ty.Iterable[ty.Dict[str, ty.Any]]:
        logger.info(f"Retrieving {k} documents for prompt: {prompt}")
        embedding = np.array([self.embedding_model.encode(prompt)])
        _, faiss_ids = self.faiss_index.search(embedding, k=k)
        for faiss_id in faiss_ids.tolist()[0]:
            yield self.fetch_document_by_id(faiss_id)

    def fetch_document_by_id(self, faiss_id: int) -> ty.Dict[str, ty.Any]:
        logger.info(f"Fetching document with faiss_id: {faiss_id}")
        return self.mongo_collection.find({"faiss_id": faiss_id})[0]


def retrieve_papers(
    prompt: str,
    embedding_model: SentenceTransformer,
    faiss_index: faiss.Index,
    mongo_collection: Collection,
    k: int,
) -> list[dict[any]]:

    retriever = Retriever(
        embedding_model=embedding_model,
        faiss_index=faiss_index,
        mongo_collection=mongo_collection,
    )

    return [doc for doc in retriever.retrieve(prompt=prompt, k=k)]


def setup_sentence_transformer(embedding_model: str) -> SentenceTransformer:
    try:
        return SentenceTransformer(embedding_model)
    except Exception as e:
        raise ValueError(f"Couldn't load model. {e}")


def setup_faiss_index(faiss_path: str) -> faiss.Index:
    try:
        return faiss.read_index(faiss_path)
    except Exception as e:
        raise ValueError(f"Couldn't load faiss index. {e}")


def setup_mongo_collection(mongo_con_str: str, mongo_db_col: str) -> Collection:
    try:
        mongo_client = MongoClient(mongo_con_str)
        database, collection = mongo_db_col.split(":")
        return mongo_client[database][collection]
    except Exception as e:
        raise ValueError(f"Couldn't load mongo collection. {e}")


def setup_retriever_params(
    embedding_model: str, faiss_path: str, mongo_con_str: str, mongo_db_col: str
) -> ty.Tuple[SentenceTransformer, faiss.Index, Collection]:
    embedding_model = setup_sentence_transformer(embedding_model)
    faiss_index = setup_faiss_index(faiss_path)
    mongo_collection = setup_mongo_collection(mongo_con_str, mongo_db_col)
    return embedding_model, faiss_index, mongo_collection


def get_papers_summary_dict(papers: ty.List[ty.Dict[str, ty.Any]]) -> ty.Dict[str, ty.Any]:
    return {
        f"paper_{i}": {
            "title": doc["title"],
            "abstract": doc["abstract"],
            "url": f"https://arxiv.org/abs/{doc["id"]}",
        }
        for i, doc in enumerate(papers)
    }


def main() -> None:
    """Runs the script."""
    parser = argparse.ArgumentParser(
        description="Run processing pipeline to save embeddings and mongodb."
    )
    parser.add_argument("-p", help=f"Prompt to query Arxiv documents")
    parser.add_argument("-i", help=f"Faiss index to use in retrieval")
    parser.add_argument("-k", help=f"Num docs to retrieve")
    parser.add_argument(
        "--log",
        default="WARNING",
        help=f"Set the logging level. Available options: {list(logging._nameToLevel.keys())}",
    )

    args = parser.parse_args()

    setup_logging(level=args.log, logger=logger)

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    faiss_index = faiss.read_index(args.i)
    mongo_client = MongoClient("mongodb://localhost:27017/")
    mongo_collection = mongo_client["arxivdb"]["arxivcol"]

    papers = retrieve_papers(
        prompt=args.p,
        embedding_model=embedding_model,
        faiss_index=faiss_index,
        mongo_collection=mongo_collection,
        k=int(args.k),
    )
    print("\n")
    for i, doc in enumerate(papers):
        print(
            f"{i+1}. {doc["title"]}\n\n {doc["abstract"][:250]}\n https://arxiv.org/abs/{doc["id"]}\n"
        )


if __name__ == "__main__":
    main()
