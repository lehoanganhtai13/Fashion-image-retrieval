# Define the paths to the docker-compose files
MILVUS_DIR=./milvus
MINIO_DIR=./minio
ETL_DIR=./etl
MODEL_DIR=./model_serving
APP_DIR=./app
STREAMLIT_DIR=./streamlit
OBSERVABILITY_DIR=./observability
NETWORK_NAME=text_to_image_retrieval_network

# Define the default target
.PHONY: setup up-server down restart start stop \
	inspect-network create-network \
	setup-volumes \
	load-dataset

# Target to all services
up-minio:
	@echo "Starting Minio service..."
	@docker compose -f $(MINIO_DIR)/docker-compose.yaml up -d

up-milvus:
	@echo "Starting Milvus service..."
	@docker compose -f $(MILVUS_DIR)/docker-compose.yaml up -d

up-model:
	@echo "Starting Model Serving service..."
	@docker compose -f $(MODEL_DIR)/docker-compose.yaml up -d --build

up-app:
	@echo "Starting App service..."
	@docker compose -f $(APP_DIR)/docker-compose.yaml up -d --build

up-streamlit:
	@echo "Starting Streamlit service..."
	@docker compose -f $(STREAMLIT_DIR)/docker-compose.yaml up -d --build

up-obs:
	@echo "Starting Observability service..."
	@docker compose -f $(OBSERVABILITY_DIR)/docker-compose.yaml up -d --build

down-minio:
	@echo "Stopping Minio service..."
	@docker compose -f $(MINIO_DIR)/docker-compose.yaml down

down-milvus:
	@echo "Stopping Milvus service..."
	@docker compose -f $(MILVUS_DIR)/docker-compose.yaml down

down-model:
	@echo "Stopping Model Serving service..."
	@docker compose -f $(MODEL_DIR)/docker-compose.yaml down

down-app:
	@echo "Stopping App service..."
	@docker compose -f $(APP_DIR)/docker-compose.yaml down

down-streamlit:
	@echo "Stopping Streamlit service..."
	@docker compose -f $(STREAMLIT_DIR)/docker-compose.yaml down

down-obs:
	@echo "Stopping Observability service..."
	@docker compose -f $(OBSERVABILITY_DIR)/docker-compose.yaml down

setup:
	@echo "Starting services..."
	@$(MAKE) up-minio
	@$(MAKE) up-milvus
	@$(MAKE) up-model

up-server:
	@echo "Starting the server..."
	@$(MAKE) up-app
	@$(MAKE) up-streamlit

up-observability:
	@$(MAKE) up-obs

destroy:
	@echo "Destroying and cleaning up services..."
	@$(MAKE) down
	@$(MAKE) clean

down:
	@echo "Stopping services..."
	@$(MAKE) down-minio
	@$(MAKE) down-milvus
	@$(MAKE) down-model
	@$(MAKE) down-app
	@$(MAKE) down-streamlit
	@$(MAKE) down-obs

stop:
	@echo "Stopping the server..."
	@$(MAKE) down-app
	@$(MAKE) down-streamlit

restart:
	@echo "Restarting services..."
	@$(MAKE) down
	@$(MAKE) setup
	@$(MAKE) up-server
	@$(MAKE) up-observability

# Targets for network
create-network:
	@echo "Creating network $(NETWORK_NAME)..."
	@docker network create $(NETWORK_NAME)
	@$(MAKE) update-env-milvus
	@$(MAKE) update-env-minio
	@$(MAKE) update-env-model-serving
	@$(MAKE) update-env-etl
	@$(MAKE) update-env-app
	@$(MAKE) update-env-streamlit
	@$(MAKE) update-env-observability

inspect-network:
	@echo "Network subnet: $(shell docker network inspect $(NETWORK_NAME) | grep -oP '(?<="Subnet": ")[^"]*')"

update-env-milvus:
	@echo "Updating milvus/.env file with NETWORK_SUBNET..."
	@SUBNET=$(shell docker network inspect $(NETWORK_NAME) | grep -oP '(?<="Subnet": ")[^"]*') && \
	if grep -q '^NETWORK_SUBNET=' milvus/.env; then \
		sed -i 's/^NETWORK_SUBNET=.*/NETWORK_SUBNET=$$SUBNET/' milvus/.env; \
	else \
		echo "NETWORK_SUBNET=$$SUBNET" >> milvus/.env; \
	fi

update-env-minio:
	@echo "Updating minio/.env file with NETWORK_SUBNET..."
	@SUBNET=$(shell docker network inspect $(NETWORK_NAME) | grep -oP '(?<="Subnet": ")[^"]*') && \
	if grep -q '^NETWORK_SUBNET=' minio/.env; then \
		sed -i 's/^NETWORK_SUBNET=.*/NETWORK_SUBNET=$$SUBNET/' minio/.env; \
	else \
		echo "NETWORK_SUBNET=$$SUBNET" >> minio/.env; \
	fi

update-env-model-serving:
	@echo "Updating model_serving/.env file with NETWORK_SUBNET..."
	@SUBNET=$(shell docker network inspect $(NETWORK_NAME) | grep -oP '(?<="Subnet": ")[^"]*') && \
	if grep -q '^NETWORK_SUBNET=' model_serving/.env; then \
		sed -i 's/^NETWORK_SUBNET=.*/NETWORK_SUBNET=$$SUBNET/' model_serving/.env; \
	else \
		echo "NETWORK_SUBNET=$$SUBNET" >> model_serving/.env; \
	fi

update-env-etl:
	@echo "Updating etl/.env file with NETWORK_SUBNET..."
	@SUBNET=$(shell docker network inspect $(NETWORK_NAME) | grep -oP '(?<="Subnet": ")[^"]*') && \
	if grep -q '^NETWORK_SUBNET=' etl/.env; then \
		sed -i 's/^NETWORK_SUBNET=.*/NETWORK_SUBNET=$$SUBNET/' etl/.env; \
	else \
		echo "NETWORK_SUBNET=$$SUBNET" >> etl/.env; \
	fi

update-env-app:
	@echo "Updating app/.env file with NETWORK_SUBNET..."
	@SUBNET=$(shell docker network inspect $(NETWORK_NAME) | grep -oP '(?<="Subnet": ")[^"]*') && \
	if grep -q '^NETWORK_SUBNET=' app/.env; then \
		sed -i 's/^NETWORK_SUBNET=.*/NETWORK_SUBNET=$$SUBNET/' app/.env; \
	else \
		echo "NETWORK_SUBNET=$$SUBNET" >> app/.env; \
	fi

update-env-streamlit:
	@echo "Updating streamlit/.env file with NETWORK_SUBNET..."
	@SUBNET=$(shell docker network inspect $(NETWORK_NAME) | grep -oP '(?<="Subnet": ")[^"]*') && \
	if grep -q '^NETWORK_SUBNET=' streamlit/.env; then \
		sed -i 's/^NETWORK_SUBNET=.*/NETWORK_SUBNET=$$SUBNET/' streamlit/.env; \
	else \
		echo "NETWORK_SUBNET=$$SUBNET" >> streamlit/.env; \
	fi

update-env-observability:
	@echo "Updating observability/.env file with NETWORK_SUBNET..."
	@SUBNET=$(shell docker network inspect $(NETWORK_NAME) | grep -oP '(?<="Subnet": ")[^"]*') && \
	if grep -q '^NETWORK_SUBNET=' observability/.env; then \
		sed -i 's/^NETWORK_SUBNET=.*/NETWORK_SUBNET=$$SUBNET/' observability/.env; \
	else \
		echo "NETWORK_SUBNET=$$SUBNET" >> observability/.env; \
	fi

# Target to setup volumes folder
setup-volumes: 
	@$(MAKE) setup-volumes-minio
	@$(MAKE) setup-volumes-milvus
	@$(MAKE) setup-volumes-model
	@$(MAKE) setup-volumes-app
	@$(MAKE) setup-volumes-observability

setup-volumes-minio:
	@echo "Creating volumes folder for minio..."
	@mkdir -p minio/.data
	@echo "Setting permissions..."
	@chmod -R 767 minio/.data/

setup-volumes-milvus:
	@echo "Creating volumes folder for milvus..."
	@mkdir -p milvus/.data/etcd milvus/.data/milvus
	@echo "Setting permissions..."
	@chmod -R 767 milvus/.data/

setup-volumes-model:
	@echo "Creating volumes folder for models..."
	@mkdir -p model_serving/clip/prometheus_multiproc model_serving/clip/logs \
		model_serving/embedder/prometheus_multiproc model_serving/embedder/logs \
		model_serving/llm/prometheus_multiproc model_serving/llm/logs \
		model_serving/reranker/prometheus_multiproc model_serving/reranker/logs
	@echo "Setting permissions..."
	@chmod -R 767 model_serving/clip/prometheus_multiproc model_serving/clip/logs \
		model_serving/embedder/prometheus_multiproc model_serving/embedder/logs \
		model_serving/llm/prometheus_multiproc model_serving/llm/logs \
		model_serving/reranker/prometheus_multiproc model_serving/reranker/logs

setup-volumes-app:
	@echo "Creating volumes folder for app..."
	@mkdir -p app/prometheus_multiproc app/logs
	@echo "Setting permissions..."
	@chmod -R 767 app/prometheus_multiproc app/logs

setup-volumes-observability:
	@echo "Creating volumes folder for observability..."
	@mkdir -p observability/.data/prometheus observability/.data/alertmanager observability/.data/grafana \
		observability/.data/elasticsearch observability/.data/fluentbit
	@echo "Setting permissions..."
	@chmod -R 767 observability/.data/ observability/config/

# Target to prepare dataset
load-dataset:
	@echo "Preparing dataset..."
	@ROOT_DIR=$(shell pwd)
	@if [ "$(load)" = "1" ]; then LOAD_FLAG="--load_only"; else LOAD_FLAG=""; fi; \
	cd $(ETL_DIR) && \
	python jobs.py --url https://huggingface.co/datasets/glami/glami-1m/resolve/main/GLAMI-1M-dataset--test-only.zip \
					--folder_name GLAMI-1M-test-dataset \
					--test \
					--batch_size 1000 \
					--overwrite \
					$$LOAD_FLAG && \
	cd $(ROOT_DIR)

load-dataset-image:
	@echo "Preparing dataset..."
	@ROOT_DIR=$(shell pwd)
	@if [ "$(load)" = "1" ]; then LOAD_FLAG="--load_only"; else LOAD_FLAG=""; fi; \
	cd $(ETL_DIR) && \
	python jobs.py --url https://huggingface.co/datasets/glami/glami-1m/resolve/main/GLAMI-1M-dataset--test-only.zip \
					--folder_name GLAMI-1M-test-dataset \
					--test \
					--batch_size 1000 \
					--overwrite \
					--collection_type image \
					$$LOAD_FLAG && \
	cd $(ROOT_DIR)

load-dataset-name:
	@echo "Preparing dataset..."
	@ROOT_DIR=$(shell pwd)
	@if [ "$(load)" = "1" ]; then LOAD_FLAG="--load_only"; else LOAD_FLAG=""; fi; \
	cd $(ETL_DIR) && \
	python jobs.py --url https://huggingface.co/datasets/glami/glami-1m/resolve/main/GLAMI-1M-dataset--test-only.zip \
					--folder_name GLAMI-1M-test-dataset \
					--test \
					--batch_size 1000 \
					--overwrite \
					--collection_type name \
					$$LOAD_FLAG && \
	cd $(ROOT_DIR)

load-dataset-category:
	@echo "Preparing dataset..."
	@ROOT_DIR=$(shell pwd)
	@if [ "$(load)" = "1" ]; then LOAD_FLAG="--load_only"; else LOAD_FLAG=""; fi; \
	cd $(ETL_DIR) && \
	python jobs.py --url https://huggingface.co/datasets/glami/glami-1m/resolve/main/GLAMI-1M-dataset--test-only.zip \
					--folder_name GLAMI-1M-test-dataset \
					--test \
					--batch_size 1000 \
					--overwrite \
					--collection_type category \
					$$LOAD_FLAG && \
	cd $(ROOT_DIR)

# Target to clean up the volumes
clean:
	@echo "Cleaning up..."
	@$(MAKE) clean-milvus
	@$(MAKE) clean-minio
	@$(MAKE) clean-model
	@$(MAKE) clean-app
	@$(MAKE) clean-observability

clean-milvus:
	@echo "Cleaning up Milvus volumes..."
	@docker run --rm -v $(MILVUS_DIR):/data alpine sh -c "rm -rf /data/.data"

clean-minio:
	@echo "Cleaning up Minio volumes..."
	@docker run --rm -v $(MINIO_DIR):/data alpine sh -c "rm -rf /data/.data"

clean-model:
	@echo "Cleaning up Model Serving volumes..."
	@docker run --rm -v $(MODEL_DIR)/clip:/data alpine sh -c "rm -rf /data/logs"
	@docker run --rm -v $(MODEL_DIR)/clip:/data alpine sh -c "rm -rf /data/prometheus_multiproc"
	@docker run --rm -v $(MODEL_DIR)/embedder:/data alpine sh -c "rm -rf /data/logs"
	@docker run --rm -v $(MODEL_DIR)/embedder:/data alpine sh -c "rm -rf /data/prometheus_multiproc"
	@docker run --rm -v $(MODEL_DIR)/llm:/data alpine sh -c "rm -rf /data/logs"
	@docker run --rm -v $(MODEL_DIR)/llm:/data alpine sh -c "rm -rf /data/prometheus_multiproc"
	@docker run --rm -v $(MODEL_DIR)/reranker:/data alpine sh -c "rm -rf /data/logs"
	@docker run --rm -v $(MODEL_DIR)/reranker:/data alpine sh -c "rm -rf /data/prometheus_multiproc"

clean-app:
	@echo "Cleaning up App volumes..."
	@docker run --rm -v $(APP_DIR):/data alpine sh -c "rm -rf /data/logs"
	@docker run --rm -v $(APP_DIR):/data alpine sh -c "rm -rf /data/prometheus_multiproc"

clean-observability:
	@echo "Cleaning up Observability volumes..."
	@docker run --rm -v $(OBSERVABILITY_DIR):/data alpine sh -c "rm -rf /data/.data"