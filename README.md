# ACIS Insurance Analytics Project

This repository contains two tasks for the ACIS Insurance Analytics project:

## Task 1: EDA and Statistical Analysis
Located in the `task-1` directory, this task includes:
- Exploratory Data Analysis (EDA)
- Statistical Analysis
- Data Visualization
- CI/CD Pipeline Setup
- Unit Tests

## Task 2: Data Version Control
Located in the `task-2` directory, this task focuses on:
- Setting up DVC for data versioning
- Creating reproducible data pipelines
- Establishing data audit trails

## Setup

Each task has its own setup instructions in its respective README.md file:
- [Task 1 Setup](task-1/README.md)
- [Task 2 Setup](task-2/README.md)

## Project Structure

```
acis-insurance-analytics/
├── task-1/             # EDA and Statistical Analysis
│   ├── src/           # Source code
│   ├── tests/         # Unit tests
│   ├── notebooks/     # Jupyter notebooks
│   ├── reports/       # Analysis outputs
│   └── .github/       # CI/CD configuration
│
├── task-2/            # Data Version Control
│   ├── data/         # DVC-tracked data
│   ├── src/          # DVC pipeline code
│   └── tests/        # DVC tests
│
└── venv/              # Virtual environment
``` 