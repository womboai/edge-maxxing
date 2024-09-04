FROM pytorch:2.2.0-cuda12.1-cudnn8-ru

WORKDIR /api

COPY neuron .
COPY pipelines .
COPY validator .

WORKDIR /api/validator/submission_tester/

RUN setup.sh

USER api

EXPOSE 8000

ENTRYPOINT ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "main:app"]
