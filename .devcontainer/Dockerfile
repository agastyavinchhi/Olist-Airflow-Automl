# Dev container image: Python 3.12 (Microsoft devcontainers base) + psql + make.
FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# The base image ships a Yarn apt source whose signing key has expired, which makes every
# `apt-get update` fail (including the one in the docker-outside-of-docker feature).
# Yarn isn't needed here (the frontend uses npm), so drop the source first.
RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends postgresql-client make \
    && rm -rf /var/lib/apt/lists/*
