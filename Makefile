project_name := jvs_data_analysis
requirements_file := requirements.txt
environment_file := environment.yml
python_version := 3.10.0

.PHONY: conda-req-import
conda-req-import:
	conda install -f $(requirements_file)

.PHONY: conda-req-export
conda-req-export:
	conda list --export > $(requirements_file)

.PHONY: conda-create
conda-create:
	conda create --name $(project_name) python=$(python_version)

.PHONY: conda-create-from-file
conda-create-from-file:
	conda create --name $(project_name) --file $(requirements_file)

.PHONY: conda-env-create
conda-env-create:
	conda env create -f $(environment_file)

.PHONY: conda-env-export
conda-env-export:
	conda env export > $(environment_file)

.PHONY: test
test:
	pytest -v

.PHONY: test-full
test-full:
	pytest -v --slow --cov=src --cov-report=html

.PHONY: format
format:
	ruff check src --fix-only --exit-zero
	black src
	isort src

.PHONY: lint
lint:
	ruff check src
	isort --check-only src
