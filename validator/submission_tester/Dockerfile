FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /api

COPY validator/api/update.sh validator/api/
RUN cd validator && ./submission_tester/update.sh
USER api

COPY pipelines .
COPY base .
COPY validator .
COPY .git .git

WORKDIR /api/validator

USER root
EXPOSE 8000
ENTRYPOINT ["submission_tester/entrypoint.sh"]
