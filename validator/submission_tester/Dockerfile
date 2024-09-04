FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get -y install pipx && pipx install poetry

WORKDIR /api

COPY neuron .
COPY pipelines .
COPY validator .

WORKDIR /api/validator/submission_tester

RUN ./setup.sh

WORKDIR /api/validator

RUN /root/.local/bin/poetry install

USER api

EXPOSE 8000

ENTRYPOINT ["poetry", "run", "uvicorn", "--host", "0.0.0.0", "--port", "8000", "submission_tester:app"]
