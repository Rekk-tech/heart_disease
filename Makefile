# Heart Disease Prediction Project Makefile

# Variables
PYTHON = python
PIP = pip
DATA_DIR = data
MODELS_DIR = models
NOTEBOOKS_DIR = notebooks
SRC_DIR = src
APP_DIR = app
TESTS_DIR = tests

# Default target
.PHONY: help
help:
	@echo "Heart Disease Prediction Project"
	@echo "================================"
	@echo ""
	@echo "Available commands:"
	@echo "  setup          - Install dependencies and setup environment"
	@echo "  data           - Load and validate data"
	@echo "  preprocess     - Preprocess data and create train/val/test splits"
	@echo "  train          - Train models and create ensemble"
	@echo "  evaluate       - Evaluate models and generate reports"
	@echo "  explain        - Generate model explanations"
	@echo "  api            - Start FastAPI server"
	@echo "  ui             - Start Streamlit UI"
	@echo "  test           - Run all tests"
	@echo "  test-coverage  - Run tests with coverage report"
	@echo "  lint           - Run linting checks"
	@echo "  clean          - Clean temporary files"
	@echo "  clean-all      - Clean all generated files"
	@echo "  pipeline       - Run complete ML pipeline"
	@echo "  docs           - Generate documentation"
	@echo ""

# Setup and Installation
.PHONY: setup
setup:
	@echo "Setting up Heart Disease Prediction Project..."
	$(PIP) install -r requirements.txt
	@echo "Setup completed!"

.PHONY: setup-dev
setup-dev: setup
	@echo "Setting up development environment..."
	$(PIP) install pytest pytest-cov black flake8 mypy
	@echo "Development setup completed!"

# Data Operations
.PHONY: data
data:
	@echo "Loading and validating data..."
	$(PYTHON) -c "from src.data_loader import load_data, validate_schema; df = load_data('$(DATA_DIR)/raw/Heart_disease_cleveland_new.csv'); print('Data validation:', 'PASSED' if validate_schema(df) else 'FAILED')"

.PHONY: preprocess
preprocess:
	@echo "Preprocessing data..."
	$(PYTHON) -c "from run_pipeline import preprocess_data; preprocess_data()"

# Model Training and Evaluation
.PHONY: train
train:
	@echo "Training models..."
	$(PYTHON) -c "from run_pipeline import train_models; train_models()"

.PHONY: evaluate
evaluate:
	@echo "Evaluating models..."
	$(PYTHON) -c "from run_pipeline import evaluate_models; evaluate_models()"

.PHONY: explain
explain:
	@echo "Generating model explanations..."
	$(PYTHON) -c "from run_pipeline import explain_models; explain_models()"

# Application Services
.PHONY: api
api:
	@echo "Starting FastAPI server..."
	cd $(APP_DIR) && $(PYTHON) api.py

.PHONY: ui
ui:
	@echo "Starting Streamlit UI..."
	streamlit run $(APP_DIR)/ui_streamlit.py

# Testing
.PHONY: test
test:
	@echo "Running tests..."
	$(PYTHON) -m pytest $(TESTS_DIR)/ -v

.PHONY: test-coverage
test-coverage:
	@echo "Running tests with coverage..."
	$(PYTHON) -m pytest $(TESTS_DIR)/ --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing

.PHONY: test-api
test-api:
	@echo "Testing API endpoints..."
	$(PYTHON) -m pytest $(TESTS_DIR)/test_api.py -v

# Code Quality
.PHONY: lint
lint:
	@echo "Running linting checks..."
	flake8 $(SRC_DIR)/ $(APP_DIR)/ $(TESTS_DIR)/
	black --check $(SRC_DIR)/ $(APP_DIR)/ $(TESTS_DIR)/

.PHONY: format
format:
	@echo "Formatting code..."
	black $(SRC_DIR)/ $(APP_DIR)/ $(TESTS_DIR)/

.PHONY: type-check
type-check:
	@echo "Running type checks..."
	mypy $(SRC_DIR)/ $(APP_DIR)/

# Complete Pipeline
.PHONY: pipeline
pipeline: setup data preprocess train evaluate explain
	@echo "Complete ML pipeline executed successfully!"

.PHONY: pipeline-quick
pipeline-quick: setup data preprocess train evaluate
	@echo "Quick ML pipeline executed successfully!"

# Jupyter Notebooks
.PHONY: notebooks
notebooks:
	@echo "Starting Jupyter notebooks..."
	jupyter notebook $(NOTEBOOKS_DIR)/

.PHONY: notebook-eda
notebook-eda:
	@echo "Opening EDA notebook..."
	jupyter notebook $(NOTEBOOKS_DIR)/01_eda.ipynb

.PHONY: notebook-viz
notebook-viz:
	@echo "Opening visualization notebook..."
	jupyter notebook $(NOTEBOOKS_DIR)/02_visualization.ipynb

.PHONY: notebook-modeling
notebook-modeling:
	@echo "Opening modeling notebook..."
	jupyter notebook $(NOTEBOOKS_DIR)/03_modeling.ipynb

.PHONY: notebook-explain
notebook-explain:
	@echo "Opening explainability notebook..."
	jupyter notebook $(NOTEBOOKS_DIR)/04_explainability.ipynb

# Documentation
.PHONY: docs
docs:
	@echo "Generating documentation..."
	@echo "Documentation would be generated here (using Sphinx or similar)"

# Cleaning
.PHONY: clean
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

.PHONY: clean-all
clean-all: clean
	@echo "Cleaning all generated files..."
	rm -rf $(MODELS_DIR)/*.pkl
	rm -rf $(MODELS_DIR)/*.json
	rm -rf $(DATA_DIR)/processed/
	rm -rf $(DATA_DIR)/interim/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/

# Docker (optional)
.PHONY: docker-build
docker-build:
	@echo "Building Docker image..."
	docker build -t heart-disease-prediction .

.PHONY: docker-run
docker-run:
	@echo "Running Docker container..."
	docker run -p 8000:8000 heart-disease-prediction

# Monitoring and Logging
.PHONY: logs
logs:
	@echo "Showing recent logs..."
	tail -f logs/heart_disease.log

# Model Management
.PHONY: model-info
model-info:
	@echo "Model information:"
	@echo "=================="
	@ls -la $(MODELS_DIR)/ || echo "No models found"

.PHONY: backup-models
backup-models:
	@echo "Backing up models..."
	tar -czf models_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz $(MODELS_DIR)/

# Environment Information
.PHONY: env-info
env-info:
	@echo "Environment Information:"
	@echo "========================"
	@echo "Python version: $$(python --version)"
	@echo "Pip version: $$(pip --version)"
	@echo "Current directory: $$(pwd)"
	@echo "Python path: $$(python -c 'import sys; print(sys.executable)')"

# Development Workflow
.PHONY: dev-setup
dev-setup: setup-dev
	@echo "Development environment ready!"
	@echo "Run 'make test' to run tests"
	@echo "Run 'make lint' to check code quality"
	@echo "Run 'make format' to format code"

.PHONY: ci
ci: lint test-coverage
	@echo "CI pipeline completed successfully!"

# Help for specific targets
.PHONY: help-data
help-data:
	@echo "Data-related commands:"
	@echo "  data        - Load and validate raw data"
	@echo "  preprocess  - Preprocess data and create splits"

.PHONY: help-models
help-models:
	@echo "Model-related commands:"
	@echo "  train       - Train all models"
	@echo "  evaluate    - Evaluate model performance"
	@echo "  explain     - Generate model explanations"

.PHONY: help-apps
help-apps:
	@echo "Application commands:"
	@echo "  api         - Start FastAPI server (localhost:8000)"
	@echo "  ui          - Start Streamlit UI (localhost:8501)"

.PHONY: help-dev
help-dev:
	@echo "Development commands:"
	@echo "  test        - Run all tests"
	@echo "  lint        - Run code linting"
	@echo "  format      - Format code with black"
	@echo "  type-check  - Run type checking with mypy"
