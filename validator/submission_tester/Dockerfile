FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y sudo pipx git git-lfs build-essential python3-dev python3-opencv

RUN useradd --create-home --home-dir /home/api api
RUN useradd --shell=/bin/false --create-home --home-dir /home/sandbox sandbox

USER api

RUN pipx install uv

WORKDIR /api

COPY --chown=api:api pipelines/pyproject.toml ./pipelines/
COPY --chown=api:api base/pyproject.toml ./base/
COPY --chown=api:api validator/pyproject.toml  validator/uv.lock ./validator/

RUN mkdir -p pipelines/pipelines
RUN mkdir -p base/base
RUN mkdir -p base/testing

WORKDIR /api/validator
RUN ~/.local/bin/uv sync --compile-bytecode --no-dev --frozen --no-install-project --no-install-workspace
WORKDIR /api

COPY --chown=api:api pipelines pipelines
COPY --chown=api:api base base
COPY --chown=api:api validator validator
COPY --chown=api:api .git .git

WORKDIR /api/validator
USER root

EXPOSE 8000

ENTRYPOINT ["submission_tester/entrypoint.sh"]
