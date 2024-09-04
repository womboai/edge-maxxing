FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /api

COPY neuron .
COPY pipelines .
COPY validator .

WORKDIR /api/validator/

RUN apt-get update && apt-get -y install pipx && pipx install poetry

RUN poetry install

WORKDIR /api/validator/submission_tester

RUN ./setup.sh

WORKDIR /api/validator

USER api

EXPOSE 8000

ENTRYPOINT ["poetry", "run", "uvicorn", "--host", "0.0.0.0", "--port", "8000", "submission_tester:app"]
