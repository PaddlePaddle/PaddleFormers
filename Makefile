# Makefile for PaddleFormers
#
# 	GitHb: https://github.com/PaddlePaddle/PaddleFormers
# 	Author: Paddle Team https://github.com/PaddlePaddle
#

.PHONY: all
all : lint test
check_dirs := paddleformers scripts tests 
# # # # # # # # # # # # # # # Format Block # # # # # # # # # # # # # # # 

format:
	pre-commit run isort
	pre-commit run black

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # Lint Block # # # # # # # # # # # # # # # 

.PHONY: lint
lint:
	$(eval modified_py_files := $(shell python scripts/codestyle/get_modified_files.py $(check_dirs)))
	@if test -n "$(modified_py_files)"; then \
		echo ${modified_py_files}; \
		pre-commit run --files ${modified_py_files}; \
	else \
		echo "No library .py files were modified"; \
	fi	

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # Test Block # # # # # # # # # # # # # # # 

.PHONY: test
test: unit-test

unit-test:
	DOWNLOAD_SOURCE=aistudio \
	PYTHONPATH=$(shell pwd) pytest -v \
		--retries 1 --retry-delay 1 \
		--durations 20 \
		--cov=./paddleformers \
		--cov-report=xml:coverage.xml

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

.PHONY: install
install:
	@echo "Checking CUDA version and selecting pip source..."
	@if ! command -v nvcc >/dev/null 2>&1; then \
	    echo "ERROR: nvcc (CUDA) not found. Please install CUDA before proceeding."; \
	    exit 1; \
	fi; \
	cuda_version=$$(nvcc --version | grep release | awk '{print $$5}' | sed 's/,//'); \
	echo "Detected CUDA version: $$cuda_version"; \
	if [ "$$cuda_version" = "12.6" ]; then \
	    PADDLE_SOURCE="https://www.paddlepaddle.org.cn/packages/nightly/cu126/"; \
	elif [ "$$cuda_version" = "12.9" ]; then \
	    PADDLE_SOURCE="https://www.paddlepaddle.org.cn/packages/nightly/cu129/"; \
	elif [ "$$cuda_version" = "13.0" ]; then \
	    PADDLE_SOURCE="https://www.paddlepaddle.org.cn/packages/nightly/cu130/"; \
	else \
	    PADDLE_SOURCE=""; \
	    echo "Unknown CUDA version."; \
	fi; \
	echo "Using pip source: $$PADDLE_SOURCE"; \
	pip install -r requirements-dev.txt \
	pip install -r requirements.txt --extra-index-url "$$PADDLE_SOURCE"; \
	pre-commit install


.PHONY: deploy-ppdiffusers
deploy-ppdiffusers:
	cd ppdiffusers && make install && make

.PHONY: deploy-paddle-pipelines
deploy-paddle-pipelines:
	cd pipelines && make install && make

.PHONY: deploy-paddleformers
deploy-paddleformers:
	# install related package
	make install
	# build
	python3 setup.py sdist bdist_wheel
	# upload
	twine upload --skip-existing dist/*
