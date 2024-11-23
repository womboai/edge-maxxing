import logging

from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource, Attributes

ENDPOINT = "http://98.81.78.238:4317"


def init_open_telemetry_logging(attributes: Attributes):
    logger_provider = LoggerProvider(resource=Resource.create(attributes=attributes))
    set_logger_provider(logger_provider)

    exporter = OTLPLogExporter(endpoint=ENDPOINT, insecure=True)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    handler = LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)

    logging.getLogger().addHandler(handler)
