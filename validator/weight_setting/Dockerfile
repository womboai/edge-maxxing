FROM python:3.10

RUN apt-get update && apt-get install -y pipx

RUN useradd --create-home --home-dir /home/validator validator
USER validator

RUN pipx install uv

WORKDIR /app/validator

COPY base/pyproject.toml base/
COPY validator/pyproject.toml validator/
RUN uv sync --compile-bytecode

COPY pipelines .
COPY base .
COPY validator .
COPY .git .git

WORKDIR /api/validator/weight_setting

USER root
ENTRYPOINT ["entrypoint.sh"]
