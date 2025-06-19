# ACIS Insurance Analytics

## Project Overview
This project analyzes insurance data to uncover patterns in risk and profitability, focusing on exploratory data analysis (EDA), statistical insights, and actionable recommendations for the business.

## Directory Structure
```
acis-insurance-analytics/
├── data/                # Raw and processed data
├── notebooks/           # Jupyter notebooks for EDA and analysis
├── src/                 # Source code (analysis, models, visualizations)
├── .github/workflows/   # CI/CD workflows
├── venv/                # Python virtual environment
```

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd acis-insurance-analytics
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the EDA notebook:
   - Open `notebooks/01_eda.ipynb` in Jupyter and run all cells.

## CI/CD
- GitHub Actions is set up to run linting and tests on every push and pull request to `main`.

## Contribution Guidelines
- Use feature branches (e.g., `task-1`, `eda-improvements`).
- Commit frequently with descriptive messages.
- Ensure code passes linting and tests before opening a pull request.

## Authors
- [Your Name]

## License
- MIT License (or specify your license) 