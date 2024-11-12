FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y curl

WORKDIR /api

COPY neuron/pyproject.toml neuron/
COPY neuron/neuron/__init__.py neuron/neuron/
COPY pipelines/pyproject.toml pipelines/
COPY pipelines/pipelines/__init__.py pipelines/pipelines/

COPY validator/pyproject.toml validator/
COPY validator/uv.lock validator/

COPY validator/base_validator/__init__.py validator/base_validator/
COPY validator/submission_tester/__init__.py validator/submission_tester/
COPY validator/weight_setting/__init__.py validator/weight_setting/

COPY validator/submission_tester/setup.sh validator/submission_tester/
COPY validator/submission_tester/update.sh validator/submission_tester/

RUN cd validator && ./submission_tester/setup.sh

COPY .git .git
RUN chown -R api:api /api/.git

COPY README.md .
COPY validator/submission_tester/start.sh validator/submission_tester/

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
    --retries=5 \
    CMD curl -f http://localhost:8000/state || exit 1

ENTRYPOINT ["submission_tester/entrypoint.sh"]
