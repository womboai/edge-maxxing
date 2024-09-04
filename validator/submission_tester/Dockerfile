FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /api

COPY neuron .
COPY pipelines .
COPY validator .

WORKDIR /api/validator/submission_tester/

RUN setup.sh

USER api

EXPOSE 8000

ENTRYPOINT ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "main:app"]
