FROM python:3.9-slim

LABEL maintainer="Tai Le"
LABEL organization="Uniquify"

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

#  Proxy used to build the image
ARG ENTRY_PROXY

WORKDIR /app

COPY requirements.txt .

ENV HTTP_PROXY="${ENTRY_PROXY}"
ENV HTTPS_PROXY="${ENTRY_PROXY}"

RUN apt-get update -o Acquire::http::Proxy="${ENTRY_PROXY}" && \
    apt-get install -y --no-install-recommends -o Acquire::http::Proxy="${ENTRY_PROXY}" && \
    apt-get clean

RUN pip install -r requirements.txt --proxy="${ENTRY_PROXY}" --no-cache-dir

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]