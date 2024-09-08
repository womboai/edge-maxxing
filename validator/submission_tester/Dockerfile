FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /api

COPY README.md .
COPY neuron neuron
COPY pipelines pipelines

COPY validator/pyproject.toml validator/
COPY validator/poetry.toml validator/
COPY validator/poetry.lock validator/
COPY validator/submission_tester/setup.sh validator/submission_tester/

RUN ./validator/submission_tester/setup.sh

COPY validator/submission_tester validator/submission_tester
COPY validator/base_validator validator/base_validator

WORKDIR /api/validator

USER api

EXPOSE 8000

ENTRYPOINT [ \
    "/home/api/.local/bin/poetry", \
    "run", \
    "uvicorn", \
    "--host", \
    "0.0.0.0", \
    "--port", \
    "8000", \
    "submission_tester:app" \
]
