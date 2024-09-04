FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-ru

RUN useradd --shell=/bin/false --no-create-home sandbox
RUN mkdir /sandbox
RUN chown sandbox:sandbox /sandbox

RUN useradd api

COPY . /api
WORKDIR /api
RUN chown api:api .

USER api

EXPOSE 8000

ENTRYPOINT ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "main:app"]
