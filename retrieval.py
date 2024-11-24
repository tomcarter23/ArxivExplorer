import faiss
import numpy as np
from pymongo import MongoClient
import argparse
import logging

from sentence_transformers import SentenceTransformer

mongo_client = MongoClient("mongodb://localhost:27017/")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class Retriever: 

    def __init__(self, embedding_model: str, faiss_index_path: str):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.faiss_index = faiss.read_index(faiss_index_path)  
        mongo_db = mongo_client["arxivdb"]
        self.mongo_collection = mongo_db["arxivcol"]

    def retrieve(self, prompt: str, k: int):
        embedding = np.array([self.embedding_model.encode(prompt)])
        _, faiss_ids = self.faiss_index.search(embedding, k=k)
        for id in faiss_ids.tolist()[0]: 
            yield self.fetch_docoment_by_id(id)

    def fetch_docoment_by_id(self, faiss_id: int):
        return self.mongo_collection.find({"faiss_id": faiss_id})[0]


def main() -> None:
    """Runs the script."""
    parser = argparse.ArgumentParser(
        description="Run processing pipeline to save embeddings and mongodb."
    )
    parser.add_argument(
        "-p", help=f"Prompt to query Arxiv documents"
    )
    parser.add_argument(
        "-k", help=f"Num docs to retrieve"
    )
    parser.add_argument(
        "--log", default="WARNING", help=f"set the logging level. Available options: {list(logging._nameToLevel.keys())}"
    )

    args = parser.parse_args()

    retriever = Retriever(embedding_model=EMBEDDING_MODEL, faiss_index_path="./faiss_index.faiss")
    print("\n")
    for i, doc in enumerate(retriever.retrieve(prompt=args.p, k=int(args.k))):
        print(f"{i+1}. {doc["title"]}\n {doc["abstract"][:250]}\n https://arxiv.org/abs/{doc["id"]}\n")


if __name__ == "__main__":
    main()




