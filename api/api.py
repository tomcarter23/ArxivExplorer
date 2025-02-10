import os

from fastapi import FastAPI, Request
import uvicorn
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from arxiv_explorer.retrieve import retrieve_papers, setup_retriever_params, get_papers_summary_dict


MONGODB_URL = os.getenv("MONGODB_URL")


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedding_model, faiss_index, mongo_collection = setup_retriever_params(
        embedding_model="all-MiniLM-L6-v2",
        faiss_path="./faiss_index.faiss",
        mongo_con_str=MONGODB_URL,
        mongo_db_col="arxivdb:arxivcol",
    )
    yield {'embedding_model': embedding_model, 'faiss_index': faiss_index, 'mongo_collection': mongo_collection}


app = FastAPI(lifespan=lifespan)


class Retrieve(BaseModel):
    query: str
    k: int = Field(default=5, gt=0)


@app.post("/retrieve")
async def root(retrieve: Retrieve, request: Request):
    papers = retrieve_papers(
        prompt=retrieve.query,
        embedding_model=request.state.embedding_model,
        faiss_index=request.state.faiss_index,
        mongo_collection=request.state.mongo_collection,
        k=retrieve.k,
    )
    papers_dict = get_papers_summary_dict(papers)
    return {"result": papers_dict}


if __name__ == "__main__":
    uvicorn.run(app, port=80, host="0.0.0.0")
