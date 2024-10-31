DATASET_URL=https://huggingface.co/datasets/glami/glami-1m/resolve/main/GLAMI-1M-dataset.zip
FOLDER_NAME=GLAMI-1M-train-dataset

.PHONY: prepare-dataset

prepare-dataset:
	@echo "Preparing dataset..."
	@python download.py --url $(DATASET_URL) --folder_name $(FOLDER_NAME)