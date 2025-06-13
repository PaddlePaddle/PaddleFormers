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
	$(eval modified_py_files := $(shell python scripts/get_modified_files.py $(check_dirs)))
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
	PYTHONPATH=$(shell pwd) pytest -v \
		-n auto \
		--retries 1 --retry-delay 1 \
		--durations 20 \
		--cov paddleformers \
		--cov-report xml:coverage.xml

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

.PHONY: install
install:
	pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
	pip install -r requirements-dev.txt
	pip install -r requirements.txt
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

.PHONY: regression-all
release: 
	bash ./scripts/regression/run_release.sh 0 0,1 all

.PHONY: regression-key
key: 
	bash ./scripts/regression/run_release.sh 0 0,1 p0
