FROM ubuntu:22.04

COPY inference /app

WORKDIR /app/inference

ENV PATH=/validator-venv/bin/:$PATH

EXPOSE 8000

ENTRYPOINT ["uvicorn", "--port", "8000", "api:app"]
