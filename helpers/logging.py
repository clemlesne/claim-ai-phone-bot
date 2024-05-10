from azure.monitor.opentelemetry import configure_azure_monitor
from helpers.config import CONFIG
from logging import Logger, getLogger, basicConfig
from opentelemetry import trace
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from os import environ


# Logging levels
_LOGGING_APP_LEVEL = CONFIG.monitoring.logging.app_level.value
_LOGGING_SYS_LEVEL = CONFIG.monitoring.logging.sys_level.value
basicConfig(level=_LOGGING_SYS_LEVEL)

# OpenTelemetry
environ["OTEL_TRACES_SAMPLER_ARG"] = str(0.5)  # Sample 50% of traces
configure_azure_monitor(
    connection_string=CONFIG.monitoring.application_insights.connection_string.get_secret_value(),
)  # Configure Azure monitor collection
AioHttpClientInstrumentor().instrument()  # Instrument aiohttp
HTTPXClientInstrumentor().instrument()  # Instrument httpx
TRACER = trace.get_tracer(
    instrumenting_library_version=CONFIG.version,
    instrumenting_module_name="com.github.clemlesne.call-center-ai",
)  # Create a tracer that will be used in the app


def build_logger(name: str) -> Logger:
    logger = getLogger(name)
    logger.setLevel(_LOGGING_APP_LEVEL)
    return logger
