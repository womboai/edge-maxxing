FROM python:3.10

RUN apt-get update && apt-get install pipx

RUN useradd --create-home --home-dir /home/validator validator

WORKDIR /app

USER validator

COPY README.md .
COPY neuron neuron
COPY pipelines pipelines

COPY validator/pyproject.toml validator/
COPY validator/poetry.toml validator/
COPY validator/poetry.lock validator/
COPY validator/submission_tester/setup.sh validator/submission_tester/

WORKDIR /app/validator

RUN pipx install poetry

COPY submission_tester submission_tester
COPY base_validator base_validator

ENV PATH="/home/validator/.local/bin:$PATH"

RUN poetry install

ENTRYPOINT [ \
    "poetry", \
    "run", \
    "start_validator" \
]
