FROM python:3.10

WORKDIR /api

COPY README.md .
COPY neuron neuron
COPY pipelines pipelines
COPY validator validator

WORKDIR /api/validator

RUN apt-get update && apt-get install pipx

RUN useradd --create-home --home-dir /home/validator validator

USER validator

RUN pipx install poetry

ENV PATH="/home/validator/.local/bin:$PATH"

RUN poetry install

ENTRYPOINT [
    "poetry",
    "run",
    "start_validator"
]
