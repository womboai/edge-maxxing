FROM python:3.10

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y sudo pipx git

RUN useradd --create-home --home-dir /home/validator validator

USER validator

RUN pipx install uv

WORKDIR /app

COPY --chown=validator:validator pipelines/pyproject.toml ./pipelines/
COPY --chown=validator:validator base/pyproject.toml ./base/
COPY --chown=validator:validator validator/pyproject.toml  validator/uv.lock ./validator/

RUN mkdir -p pipelines/pipelines
RUN mkdir -p base/base
RUN mkdir -p base/testing

WORKDIR /app/validator
RUN ~/.local/bin/uv sync --compile-bytecode --no-dev --frozen --no-install-project --no-install-workspace --prerelease=allow
WORKDIR /app

COPY --chown=validator:validator pipelines pipelines
COPY --chown=validator:validator base base
COPY --chown=validator:validator validator validator
COPY --chown=validator:validator .git .git

WORKDIR /app/validator

USER root

ENTRYPOINT ["weight_setting/entrypoint.sh"]
