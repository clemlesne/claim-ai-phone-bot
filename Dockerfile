# Base container
FROM docker.io/library/python:3.12-slim-bookworm@sha256:5dc6f84b5e97bfb0c90abfb7c55f3cacc2cb6687c8f920b64a833a2219875997 AS base

# Build container
FROM base AS build

RUN rm -f /etc/apt/apt.conf.d/docker-clean \
  && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked --mount=target=/root/.cache/pip,type=cache,sharing=locked \
  apt-get update -q \
  && apt-get install -y -q --no-install-recommends \
  gcc \
  python3-dev \
  && python3 -m pip install --upgrade \
  pip \
  setuptools \
  wheel

RUN python -m venv /venv
ENV PATH=/venv/bin:$PATH

COPY requirements.txt .
RUN --mount=target=/root/.cache/pip,type=cache,sharing=locked \
  python3 -m pip install --requirement requirements.txt

# Output container
FROM base

ARG VERSION
ENV VERSION=${VERSION}

RUN useradd -m appuser \
  && mkdir /app \
  && chown -R appuser:appuser /app

USER appuser

COPY --from=build /venv /venv
ENV PATH=/venv/bin:$PATH

COPY --chown=appuser:appuser . /app

WORKDIR /app

CMD ["bash", "-c", "WEB_CONCURRENCY=4 uvicorn main:api --host 0.0.0.0 --port 8080 --proxy-headers --no-server-header --timeout-keep-alive 60 --header x-version:${VERSION}"]
