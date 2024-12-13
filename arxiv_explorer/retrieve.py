import faiss
import numpy as np
from pymongo import MongoClient
from pymongo.collection import Collection
import argparse
import logging
import typing as ty

from sentence_transformers import SentenceTransformer


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

    def retrieve(self, prompt: str, k: int) -> ty.Iterable[ty.Dict[str, ty.Any]]:
        embedding = np.array([self.embedding_model.encode(prompt)])
        _, faiss_ids = self.faiss_index.search(embedding, k=k)
        for faiss_id in faiss_ids.tolist()[0]:
            yield self.fetch_document_by_id(faiss_id)

    def fetch_document_by_id(self, faiss_id: int) -> ty.Dict[str, ty.Any]:
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
