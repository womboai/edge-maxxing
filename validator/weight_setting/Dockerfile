FROM python:3.10

RUN apt-get update && apt-get install -y sudo pipx

RUN useradd --create-home --home-dir /home/validator validator

WORKDIR /app

USER validator

RUN pipx install uv

COPY neuron/pyproject.toml neuron/
COPY neuron/neuron/__init__.py neuron/neuron/
COPY pipelines/pyproject.toml pipelines/
COPY pipelines/pipelines/__init__.py pipelines/pipelines/

COPY validator/pyproject.toml validator/
COPY validator/uv.lock validator/

COPY .git .git
COPY README.md .

COPY validator/base_validator/__init__.py validator/base_validator/
COPY validator/submission_tester/__init__.py validator/submission_tester/
COPY validator/weight_setting/__init__.py validator/weight_setting/

USER root
RUN chown -R validator:validator /app
USER validator

WORKDIR /app/validator
ENV PATH="/home/validator/.local/bin:$PATH"

RUN uv sync

COPY validator/base_validator base_validator
COPY validator/weight_setting weight_setting
COPY neuron/neuron ../neuron/neuron
COPY pipelines/pipelines ../pipelines/pipelines

USER root
ENTRYPOINT ["weight_setting/entrypoint.sh"]