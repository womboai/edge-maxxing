FROM python:3.10
# pytorch/pytorch:2.2.0-cuda12.1-cudnn8-ru

RUN adduser sandbox --shell=/bin/false --no-create-home
RUN mkdir /sandbox
RUN chown sandbox:sandbox /sandbox

RUN adduser api

COPY . /api
WORKDIR /api
RUN chown api:api .

USER api

ENTRYPOINT ["/bin/bash"]
