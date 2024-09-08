FROM python:3.10

RUN apt-get update && apt-get install -y pipx

RUN useradd --create-home --home-dir /home/validator validator

WORKDIR /app

USER validator

RUN pipx install poetry

COPY README.md .
COPY neuron neuron
COPY pipelines pipelines

COPY validator/pyproject.toml validator/
COPY validator/poetry.toml validator/
COPY validator/poetry.lock validator/
COPY validator/submission_tester/setup.sh validator/submission_tester/
COPY validator/submission_tester/__init__.py validator/submission_tester
COPY validator/base_validator/__init__.py validator/base_validator

USER root
RUN chown -R validator:validator /app
USER validator

WORKDIR /app/validator
ENV PATH="/home/validator/.local/bin:$PATH"

RUN poetry install

COPY validator/submission_tester submission_tester
COPY validator/base_validator base_validator

ENTRYPOINT [ \
    "poetry", \
    "run", \
    "start_validator" \
]
