FROM python:3.9-slim

LABEL maintainer="Tai Le"
LABEL organization="Uniquify"

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

#  Proxy used to build the image
ARG ENTRY_PROXY

WORKDIR /app

COPY ../requirements.txt /app/requirements.txt

RUN apt-get update -o Acquire::http::Proxy="${ENTRY_PROXY}" && \
    apt-get install -y --no-install-recommends iputils-ping -o Acquire::http::Proxy="${ENTRY_PROXY}" && \
    apt-get clean

RUN pip install -r /app/requirements.txt --proxy="${ENTRY_PROXY}" --no-cache-dir

# Directory where source code is located
ARG SERVICE_DIR

COPY ../${SERVICE_DIR} /app
COPY ../models.py /app/models.py

EXPOSE 8000

# CMD ["python", "app.py"]