FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y curl

WORKDIR /api

COPY README.md .
COPY neuron neuron
COPY pipelines pipelines

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

WORKDIR /api/validator

EXPOSE 8000

HEALTHCHECK \
    --interval=1m \
    --timeout=5s \
    --start_period=15s \
    --start_interval=10s \
    CMD curl -f http://localhost:8000/state || exit 1

ENTRYPOINT ["submission_tester/entrypoint.sh"]
