FROM python:3.12-slim

WORKDIR /app

COPY ./pyproject.toml ./pyproject.toml
COPY ./arxiv_explorer ./arxiv_explorer
COPY ./api	./api
COPY ./output_data/faiss_index.faiss ./faiss_index.faiss

RUN pip install -e ".[api]"

ENV PYTHONPATH=./

CMD ["python", "api/api.py"]
