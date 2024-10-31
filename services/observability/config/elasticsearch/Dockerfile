FROM docker.elastic.co/elasticsearch/elasticsearch:8.15.1

LABEL maintainer="Tai Le"
LABEL organization="Uniquify"

# Proxy used to build the image
ARG ENTRY_PROXY

ENV HTTP_PROXY="${ENTRY_PROXY}"
ENV HTTPS_PROXY="${ENTRY_PROXY}"

# Install expect to handle the entrypoint script
USER root

RUN apt-get update -o Acquire::http::Proxy="${ENTRY_PROXY}" && \
    apt-get install -y expect --no-install-recommends -o Acquire::http::Proxy="${ENTRY_PROXY}" && \
    apt-get clean

COPY entrypoint.sh /usr/share/elasticsearch/entrypoint.sh

RUN chmod +x /usr/share/elasticsearch/entrypoint.sh

# Switch back to elasticsearch user
USER elasticsearch