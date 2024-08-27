FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

COPY ./inference/ /app/inference

WORKDIR /app/inference

RUN pip install -r requirements.txt

EXPOSE 8000

ENTRYPOINT ["uvicorn", "--port", "8000", "api:app"]
