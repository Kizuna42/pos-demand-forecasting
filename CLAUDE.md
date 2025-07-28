# CLAUDE.md
Always respond in Japanese

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based fresh food demand forecasting and analysis system (`pos2`) that predicts demand using time, congestion, and weather data. The system uses RandomForest models with cross-validation and includes demand curve analysis for price optimization.

## Development Commands

### Core Development Tasks
- `make requirements` - Install Python dependencies
- `make lint` - Run linting (flake8, isort, black)
- `make format` - Format code with isort and black
- `make test` - Run pytest tests
- `make clean` - Clean compiled Python files
- `make data` - Process dataset using pos2/dataset.py

### Documentation
- `mkdocs build` - Build documentation
- `mkdocs serve` - Serve documentation locally

### Environment Setup
- `make create_environment` - Create virtualenv environment
- Python version: 3.10

## Code Architecture

### Package Structure
- `pos2/` - Main Python package
  - `config.py` - Project paths and logging configuration
  - `dataset.py` - Data processing CLI with typer
  - `features.py` - Feature engineering
  - `modeling/` - ML models (train.py, predict.py)
  - `plots.py` - Visualization utilities

### Key Directories
- `data/raw/` - Raw CSV data (Shift-JIS encoded)
- `data/processed/` - Processed datasets
- `notebooks/` - Jupyter analysis notebooks
- `models/` - Saved ML models
- `reports/figures/` - Generated visualizations

### Data Format
Input data is Shift-JIS encoded CSV with columns: 商品コード, 商品名称, 年月日, 金額, 数量, 平均価格

## Code Style Configuration

- **Black**: Line length 99, excludes .git and .venv
- **Isort**: Black profile, known_first_party = ["pos2"]
- **Flake8**: Max line length 99, ignores E731,E266,E501,C901,W503, excludes notebooks,references,models,data

## Dependencies

Key libraries: loguru, typer, tqdm, python-dotenv, mkdocs, pytest, black, flake8, isort

## Development Notes

- Uses cookiecutter data science project structure
- Logging configured with loguru and tqdm integration
- CLI tools built with typer
- Quality assessment: Premium (R²≥0.7), Standard (0.5-0.7), Basic (0.3-0.5), Rejected (<0.3)