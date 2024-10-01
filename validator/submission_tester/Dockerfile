FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y curl

WORKDIR /api

COPY README.md .
COPY neuron/pyproject.toml neuron/
COPY neuron/poetry.lock neuron/
COPY neuron/neuron/__init__.py neuron/
COPY pipelines/pyproject.toml pipelines/
COPY pipelines/poetry.lock pipelines/
COPY pipelines/pipelines/__init__.py pipelines/

COPY validator/pyproject.toml validator/
COPY validator/poetry.toml validator/
COPY validator/poetry.lock validator/

COPY validator/base_validator/__init__.py validator/base_validator/
COPY validator/submission_tester/__init__.py validator/submission_tester/
COPY validator/weight_setting/__init__.py validator/weight_setting/

COPY validator/submission_tester/setup.sh validator/submission_tester/

RUN ./validator/submission_tester/setup.sh

COPY validator/base_validator validator/base_validator
COPY validator/submission_tester validator/submission_tester
COPY neuron/neuron neuron/neuron
COPY pipelines/pipelines pipelines/pipelines

WORKDIR /api/validator

EXPOSE 8000

HEALTHCHECK \
    --interval=1m \
    --timeout=5s \
    --start-period=15s \
    --start-interval=10s \
    CMD curl -f http://localhost:8000/state || exit 1

ENTRYPOINT ["submission_tester/entrypoint.sh"]
