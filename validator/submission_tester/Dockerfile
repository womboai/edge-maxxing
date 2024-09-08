FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /api

COPY README.md .
COPY neuron neuron
COPY pipelines pipelines
COPY validator validator

RUN ./validator/submission_tester/setup.sh

WORKDIR /api/validator

USER api

EXPOSE 8000

ENTRYPOINT [ \
    "/home/api/.local/bin/poetry", \
    "run", \
    "uvicorn", \
    "--host", \
    "0.0.0.0", \
    "--port", \
    "8000", \
    "submission_tester:app" \
]
